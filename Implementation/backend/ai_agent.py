from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv
from notion_client_manager import NotionManager
import os, json, traceback
import asyncio
import threading

class Violation(BaseModel):
    # https://ai.google.dev/gemini-api/docs/structured-output?lang=python
        text: str
        categories: list[str]
        severity: str
        location: str
        explanation: str

class AI_Agent:
    load_dotenv()
    def __init__(self, api_key=None):

        self.api_key = api_key or os.getenv("GEMINI_API")

        if not self.api_key:
            print(">>> Error: No Google API key set.")
        self._setup_event_loop()

        self.client = genai.Client(api_key=self.api_key)
        self.selected_model = "gemini-2.0-flash-lite"

    def _setup_event_loop(self):
        """Create and set an event loop for the current thread if one doesn't exist"""
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            # If there's no event loop in the current thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            print(f">>> Created new event loop for thread: {threading.current_thread().name}")

    def test_connection(self):
        """ Test connection to Google """
        response = self.client.models.generate_content(
            model=self.selected_model,
            contents="Explain how AI works in 20 words."
        )
        print(response.text)

    def get_selected_models(self):
        return [self.selected_model]

    def analyze_notion_using_gemini(self, content):
        try:
            # Create prompt for SOC2 analysis
            prompt = f"""
                    You are a SOC2 compliance expert. Analyze the following text content from a Notion page for any SOC2 compliance violations.
                    
                    Focus on these categories:
                    1. Security - sensitive data like passwords, API keys, tokens, credentials
                    2. Availability - information that might impact system availability
                    3. Processing Integrity - information affecting data accuracy and processing
                    4. Confidentiality - information that should remain confidential
                    5. Privacy - personally identifiable information (PII) or other private data
                    
                    Keep these instructions in mind:
                    a. Do NOT report already redacted information, especially if it has '*' masking characters.
                    b. Avoid over reporting violations, we do not want false positives.
                    
                    It is okay to not find any compliance violations. 
                    
                    Return your analysis as JSON in this format:
                    ```json
                    {{
                      "violations": [
                        {{
                          "text": "the exact text that violates compliance",
                          "categories": ["security", "availability", "processing_integrity", "confidentiality", "privacy",
                            "compliant" ],
                          "severity": "high",
                          "location" : "the Notion page id (the key of each entry)",
                          "explanation": "brief explanation of why, 20 words or less"
                        }}
                      ],
                      "compliant": false,
                      "summary": "summary of findings"
                    }}
                    Here's the content to analyze:

                    {content}
                    """

            # Send to Gemini model
            response = self.client.models.generate_content(
                model=self.selected_model,
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': list[Violation],
                },
            )

            json_response = response.text

            found_violations: list[Violation] = response.parsed
            return json_response

        except Exception as e:
            print(f">>> Error analyzing content: {e}")
            return f"Analysis failed: {str(e)}"

    def suggest_remediation_actions(self, violations):
        """ Ask AI to give specific recommendations for remediation actions for identified violations"""
        print(">>> Starting AI Agent Remediation Suggestions")
        try:
            if isinstance(violations, str):
                try:
                    violations_data = json.loads(violations)
                except Exception as e:
                    print(f">>> Error parsing JSON: {e}")
                    return []
            else:
                violations_data = violations

            violations_list = violations_data

            if not violations_list:
                print(">>> No Violations found to process")
                return []
            remediation_actions = []

            for idx, violation in enumerate(violations_list):
                prompt=f""""
                    As a SOC2 compliance expert, provide a remediation action for this violation:
            
                    Violation Text: {violation['text']}
                    Categories: {', '.join(violation['categories'])}
                    Severity: {violation['severity']}
                        
                    Return your remediation as a JSON object with these exact fields:
                    {{
                    "sensitive_text": "the exact sensitive text that should be redacted",
                    "replacement_text": "the text that should replace the sensitive text",
                    "explanation": "brief explanation of why this remediation addresses the issue, 10 words or less"
                    }}
                        
                    Only identify specific sensitive data (like credentials, PII, etc.) for remediation.
                    The sensitive_text must be a substring of the violation text.
                    The replacement_text must be of similar length but safely masked (e.g., "********" for passwords).
                    DO NOT select text that is already masked with asterisks (*) for redaction.
                    Focus only on finding actual sensitive data that needs to be masked.
                """
                try:
                    response = self.client.models.generate_content(
                        model=self.selected_model,
                        contents=prompt,
                        config={
                            'response_mime_type': 'application/json',
                        }
                    )
                    resp_data = json.loads(response.text)
                    suggestion = resp_data[0]

                    # Add the original violation to the suggestion
                    suggestion['violation'] = violation

                    remediation_actions.append(suggestion)

                except Exception as e:
                    print(f">>> Error processing remediation for violation {idx + 1}: {e}")

            return remediation_actions

        except Exception as e:
            print(f">>> Error from AI while generating remediation suggestions: {e}")
            traceback.print_exc()
            return []

    def apply_ai_remediation_suggestions(self, ai_suggestions, notion_manager=None):
        """ Apply AI Remediation Suggestions to Notion workspace """
        print(">>> Applying AI Remediation Suggestions to Notion workspace")

        application_results = {
            "total_suggested_remediations": len(ai_suggestions),
            "successful_remediations": 0,
            "failed_remediations": 0,
            "details": []
        }

        try:
            if notion_manager is None:
                from notion_client_manager import NotionManager
                notion_manager = NotionManager(api_key=os.getenv("NOTION_API_KEY"))

            for idx, suggestion in enumerate(ai_suggestions):

                try:
                    sensitive_text = suggestion.get('sensitive_text')
                    replacement_text = suggestion.get('replacement_text')
                    violation = suggestion.get('violation', {}) # this is a dict
                    location = violation.get('location')

                    remediation_detail = {
                        "sensitive_text": sensitive_text,
                        "replacement_text": replacement_text,
                        "location": location,
                        "status": "pending"
                    }

                    if not sensitive_text or not replacement_text or not violation:
                        print(">>> No Violation found to process... Skipping...")
                        continue

                    print(f">>> Attempting to redact: '{sensitive_text}' with '{replacement_text}'")
                    if sensitive_text == replacement_text:
                        print(">>> WARNING: Sensitive text and replacement text are identical!")
                    if all(c == '*' for c in sensitive_text):
                        print(">>> WARNING: Sensitive text appears to be already redacted!")

                    block_info = notion_manager._find_block_with_text(location, sensitive_text)

                    if not block_info:
                        print(">>> Unable to locate block information for Notion... Skipping...")
                        if len(sensitive_text) > 30:
                            shorter_text = sensitive_text[:30]
                            print(f">>> Trying with shorter text: '{shorter_text}'")
                            block_info = notion_manager._find_block_with_text(location, shorter_text)
                        continue

                    block_id = block_info['block_id']
                    block_type = block_info['block_type']

                    success = notion_manager.update_block_with_redacted_text(block_id, block_type, sensitive_text, replacement_text)

                    if success:
                        print(">>> Successfully redacted sensitive data in Notion with AI!")
                        remediation_detail["status"] = "completed"
                        remediation_detail["block_id"] = block_id
                        remediation_detail["block_type"] = block_type
                        application_results["successful_remediations"] += 1
                    else:
                        print(">>> Failed to redact sensitive data in Notion with AI!")
                        remediation_detail["status"] = "failed"
                        remediation_detail["block_id"] = block_id
                        remediation_detail["block_type"] = block_type
                        application_results["failed_remediations"] += 1

                    application_results["details"].append(remediation_detail)

                except Exception as e:
                    print(f">>> Error processing remediation suggestion for violation {idx + 1}: {e}")

            print(f">>> AI Remediation Suggestions applied: {application_results['successful_remediations']}/{application_results['total_suggested_remediations']} updates successful")
            return application_results
        except Exception as e:
            print(f">>> Error applying AI Remediation Suggestions: {e}")

