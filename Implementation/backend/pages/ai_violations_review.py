import dash
from dash.exceptions import PreventUpdate
import json
from dash import html, dcc, callback, Input, Output, ctx, ALL, no_update, State
import dash_bootstrap_components as dbc
from ai_agent import AI_Agent
from notion_client_manager import NotionManager
from encryption_manager import EncryptionManager
from database_manager import DatabaseManager

dash.register_page(__name__)
def create_remediate_modal():
    """ Create pop up model for when user wants to remediate violation"""
    return dbc.Modal([
        dbc.ModalHeader(html.H1("Remediate Violation")),
        dbc.ModalBody([
            html.Div(id="remediation-display"),
            html.Div(id="ai-suggestion", children=[]),
            html.Div(id="remediation-status", children=[])
        ]),
        dbc.ModalFooter([
            dbc.Button("Apply Remediation", id="apply-remediation-button", className="ml-auto")
        ]),
        ],
        id="remediation-modal",
        is_open=False,)


def create_verification_modal():
    """ Create a pop up modal for when user wants to verify AI findings in cards"""
    return dbc.Modal([
        dbc.ModalHeader(html.H1("Verify AI Findings")),
        dbc.ModalBody([
            html.Div(id="ai-violation-display"),
            html.P("Adjust the violation categories:"),
            html.Div([
                dbc.Switch(
                    value=False,
                    id={"type": "ai-verification-switch", "index": category},
                    label=category.capitalize(),
                ) for category in ["security", "availability", "processing_integrity", "confidentiality", "privacy", "compliant"]
            ]),
            html.P("Adjust severity level:"),
            dbc.Select(
                id="severity-select",
                options=[
                    {"label": "High", "value": "high"},
                    {"label": "Medium", "value": "medium"},
                    {"label": "Low", "value": "low"},
                    {"label": "Informational", "value": "info"}
                ],
                value="medium"
            ),
            html.P("Edit explanation:"),
            dbc.Textarea(
                id="explanation-textarea",
                placeholder="Enter explanation...",
                style={"width": "100%", "height": 100}
            )
        ]),
        dbc.ModalFooter([
            dbc.Button("Submit Verification", id="submit-verification", color="primary")
        ])
    ],
        id="verification-modal",
        is_open=False)

# Layout

layout = html.Div([
    dcc.Store(id="verified-api-key-store", storage_type="session"),
    dcc.Store(id="ai-analysis-results-store", data=None),
    dcc.Store(id="current-violation-store", data=None),
    dcc.Store(id="verified-findings-store", data=[]),
    dcc.Store(id="remediation-suggestions-store", data=None),
    dcc.Store(id="remediation-results-store", data=None),
    dcc.Store(id="review-id-store", data=None),

    # create display cards
    create_verification_modal(),
    create_remediate_modal(),
    dbc.Container(

        dcc.Loading(
            id="loading-review",
            children=[
                html.H1("AI Violations Review", className="mb-3"),
                html.Div(id="ai-agent-page-content"),
                dbc.Stack([
                    dbc.Button(
                            "Refresh Data",
                            id="refresh-api-button",
                            color="info",
                            className=" mb-4 mx-auto"
                        ),
                    dbc.Button(
                        "Dashboard",
                        id="ai-dashboard-button",
                        color="success",
                        className=" mb-4 mx-auto",
                        href="/dashboard"
                    )
                ]),

                dbc.Toast(
                    "",
                    id="save-result-toast",
                    header="",
                    is_open=False,
                    duration=5000,
                    icon="bi-bookmark-check",
                    className="position-fixed top-50 start-50 translate-middle",
                    color="success"
                ),
                dbc.Toast(
                    "",
                    id="remediation-toast",
                    header="",
                    is_open=False,
                    duration=5000,
                    icon="bi-shield-check",
                    className="position-fixed top-50 start-50 translate-middle",
                    color="success"
                )
            ],
            type="cube",
            className="position-fixed top-50 start-50 translate-middle"
        ),
    )
])


@callback(
    [
        Output("ai-agent-page-content", "children"),
        Output("ai-analysis-results-store", "data")
    ],
    Input("verified-api-key-store", "data"),
    State("ai-analysis-results-store", "data")
)
def load_init_data(encrypted_api_key,existing_data):
    """ Refresh data in cards displaying findings from AI Agent"""

    if not encrypted_api_key:
        return dbc.Alert(
            "No API Key Provided. Please return to home page and try again",
            color="danger",
        ), None

    # if we already called the API and have data
    if existing_data:
        encryption_manager = EncryptionManager()
        api_key = encryption_manager.decrypt_data(encrypted_api_key).decode()

        analysis_cards = []
        for violation in existing_data:
            card = create_display(violation, api_key)
            if violation.get("verified", False):
                card.children[0].children[0].children.append(
                    dbc.Badge("Verified", color="success", className="ms-2")
                )
            analysis_cards.append(card)

        return analysis_cards, existing_data

    ai_agent = AI_Agent()
    encryption_manager = EncryptionManager()
    api_key = encryption_manager.decrypt_data(encrypted_api_key).decode()
    notion_manager = NotionManager(api_key=api_key)

    all_text_string, json_results = notion_manager.return_all_extraction_results()

    # get ai analysis
    ai_analysis = ai_agent.analyze_notion_using_gemini(json_results)
    ai_analysis_json = json.loads(ai_analysis)
    analysis_cards = []
    if not ai_analysis_json:
        return dbc.Alert(
            "No violations found",
            color="success",
        )

    for violation in ai_analysis_json:
        analysis_cards.append(create_display(violation, api_key))

    return analysis_cards, ai_analysis_json


@callback(
    [
        Output("ai-analysis-results-store", "data", allow_duplicate=True),
        Output("ai-agent-page-content", "children", allow_duplicate=True),
    ],
    Input("remediation-results-store", "data"),
    [
        State("ai-analysis-results-store", "data"),
        State("verified-api-key-store", "data"),
    ],
    prevent_initial_call=True
)
def update_display_after_remediation(remediation_results, ai_results, encrypted_api_key):
    """Update the display after remediation is complete"""
    if not remediation_results or not ai_results:
        raise PreventUpdate

    # Create a copy of the AI results to avoid modifying the original
    updated_ai_results = []

    # Go through each AI result
    for result in ai_results:
        updated_result = result.copy()

        # Check if this result has been remediated
        for detail in remediation_results.get("details", []):
            # Match by text content and location
            if (detail.get("location") == result.get("location") and
                    detail.get("sensitive_text") in result.get("text", "")):
                # Mark as remediated for display
                updated_result["remediated"] = True
                updated_result["remediation_status"] = detail.get("status", "unknown")

                # Update the text to show the redacted version
                if detail.get("status") == "completed":
                    updated_result["text"] = result.get("text", "").replace(
                        detail.get("sensitive_text", ""),
                        detail.get("replacement_text", "")
                    )

        updated_ai_results.append(updated_result)

    # Recreate the display with updated data
    encryption_manager = EncryptionManager()
    api_key = encryption_manager.decrypt_data(encrypted_api_key).decode()
    analysis_cards = []

    for violation in updated_ai_results:
        card = create_display(violation, api_key)
        analysis_cards.append(card)

    return updated_ai_results, analysis_cards


def create_display(ai_analysis_data, unencrypted_api_key):
    """ Create layout display based on AI Agent analysis results"""

    category_badges = []
    for category in ai_analysis_data.get("categories", []):
        color_map = {
            "security": "danger",
            "availability": "warning",
            "processing integrity": "info",
            "confidentiality": "primary",
            "privacy": "secondary"
        }
        color = color_map.get(category.lower(), "light")
        category_badges.append(
            dbc.Badge(category.capitalize(), color=color, className="me-1")
        )

    severity = ai_analysis_data.get("severity", "").lower()
    if severity == "high":
        card_color = "danger"
    elif severity == "medium":
        card_color = "warning"
    else:
        card_color = "info"

    notion_manager = NotionManager(api_key=unencrypted_api_key)
    if ai_analysis_data.get("location"):
        url_link = notion_manager.get_page_url(ai_analysis_data.get("location"))
    else:
        url_link = "/"

    header_content = [
        html.H4(f"Severity: {severity.capitalize()}", className=f"text-{card_color}",
                style={'font-weight': 'bold'}),
        html.Div(category_badges, className="mb-2"),
    ]

    if ai_analysis_data.get("verified", False):
        header_content.append(
            dbc.Badge("Verified", color="success", pill=True, className="ms-2")
        )

    if ai_analysis_data.get("remediated", False):
        status = ai_analysis_data.get("remediation_status", "")
        badge_color = "success" if status == "completed" else "warning"
        badge_text = "Remediated" if status == "completed" else "Remediation Attempted"
        header_content.append(
            dbc.Badge(badge_text, color="purple", pill=True, className="ms-2")
        )

    violation_id = ai_analysis_data.get("text", "")[:20]

    # Create button stack based on remediation status
    button_stack = []

    # Verify Findings button
    button_stack.append(
        dbc.Button(
            "Verify Findings",
            id={"type": "verification-button", "index": violation_id},
            color="secondary",
            className="mb-4 mx-auto",
        )
    )

    # if not already successfully remediated
    if not (ai_analysis_data.get("remediated", False) and
            ai_analysis_data.get("remediation_status") == "completed"):
        button_stack.append(
            dbc.Button(
                "Remediate",
                id={"type": "remediation-button", "index": violation_id},
                color="warning",
                className="mb-4 mx-auto",
            )
        )
    return html.Div([
        dbc.Card([
            dbc.CardHeader(header_content),
            dbc.CardBody([
                dbc.CardLink("View Source Page", href=url_link, target="_blank"),
                html.P("Violation Text:", className="fw-bold"),
                dbc.Alert(ai_analysis_data.get("text", "No text provided"), color=card_color),
                html.P("Explanation:", className="fw-bold"),
                html.P(ai_analysis_data.get("explanation", "No explanation provided")),
                dbc.Stack(button_stack, className="mb-3")
            ])
        ], className="mb-3", color=card_color, outline=True)
    ])

# open modals for verifying findings

@callback(
    [
        Output("verification-modal", "is_open"),
        Output("ai-violation-display", "children"),
        Output({"type": "ai-verification-switch", "index": ALL}, "value"),
        Output("severity-select", "value"),
        Output("explanation-textarea", "value"),
        Output("current-violation-store", "data")
    ],
    [
        Input({"type": "verification-button", "index": ALL}, "n_clicks"),
        Input("submit-verification", "n_clicks"),
        Input({"type": "ai-verification-switch", "index": ALL}, "value")
    ],
    [
        State("verification-modal", "is_open"),
        State("ai-analysis-results-store", "data"),
        State({"type": "ai-verification-switch", "index": ALL}, "id")
    ],
    prevent_initial_call=True
)
def toggle_verification_modal(verification_clicks, submit_clicks,toggle_values, is_open, ai_results, toggle_ids_state):
    """Open/close the verification modal and populate it with data"""
    ctx_triggered = ctx.triggered_id

    if ctx_triggered is None:
        raise PreventUpdate

    if isinstance(ctx_triggered, dict) and ctx_triggered.get("type") == "verification-button":
        if not any(click and click > 0 for click in verification_clicks):
            raise PreventUpdate

    if ctx_triggered == "submit-verification":
        return False, [], [False] * 6, "medium", "", None

    if isinstance(ctx_triggered, dict) and ctx_triggered.get("type") == "ai-verification-switch":
        # Check if "compliant" was toggled on
        toggle_states = {toggle_id["index"]: value for toggle_id, value in zip(toggle_ids_state, toggle_values)}

        if toggle_states.get("compliant", False):
            # If compliant is toggled on, set all other toggles to False
            new_toggle_values = [False] * len(toggle_values)
            compliant_index = next(i for i, toggle_id in enumerate(toggle_ids_state)
                                   if toggle_id["index"] == "compliant")
            new_toggle_values[compliant_index] = True
            return no_update, no_update, new_toggle_values, no_update, no_update, no_update

        return is_open, no_update, toggle_values, no_update, no_update, no_update

    # Open modal when verification button is clicked
    if isinstance(ctx_triggered, dict) and ctx_triggered.get("type") == "verification-button":
        index = ctx_triggered["index"]

        # Find the violation data
        violation_data = None
        for result in ai_results:
            if result.get("text", "")[:20] == index:
                violation_data = result
                break

        if not violation_data:
            raise PreventUpdate

        categories = ["security", "availability", "processing_integrity", "confidentiality", "privacy", "compliant"]
        toggle_values = [category in violation_data.get("categories", []) for category in categories]

        display_content = [
            html.P(f"Violation Text: {violation_data.get('text', 'No text')}"),
            html.P(f"Severity: {violation_data.get('severity', 'medium')}"),
            html.P(f"Categories: {', '.join(violation_data.get('categories', []))}"),
            html.P(f"Location: {violation_data.get('location', 'Unknown')}")
        ]

        # Get severity (with a default if not present)
        severity = violation_data.get("severity", "medium")

        return True, display_content, toggle_values, severity, violation_data.get("explanation", ""), violation_data

    return is_open, [], [False] * 6, "medium", "", None


@callback(
    [
        Output("ai-analysis-results-store", "data", allow_duplicate=True),
        Output("ai-agent-page-content", "children", allow_duplicate=True),
        Output("save-result-toast", "is_open", allow_duplicate=True),
        Output("save-result-toast", "children", allow_duplicate=True),
        Output("save-result-toast", "header", allow_duplicate=True),
        Output("save-result-toast", "icon", allow_duplicate=True),
        Output("save-result-toast", "color", allow_duplicate=True),
        Output("review-id-store", "data", allow_duplicate=True)
    ],
    Input("submit-verification", "n_clicks"),
    [
        State({"type": "ai-verification-switch", "index": ALL}, "value"),
        State({"type": "ai-verification-switch", "index": ALL}, "id"),
        State("severity-select", "value"),
        State("explanation-textarea", "value"),
        State("current-violation-store", "data"),
        State("ai-analysis-results-store", "data"),
        State("verified-api-key-store", "data"),
        State("review-id-store", "data")
    ],
    prevent_initial_call=True
)
def handle_verification(n_clicks, toggle_values, toggle_ids, severity, explanation, current_violation, ai_results, encrypted_api_key, review_id):
    """Handle the submission of verification data"""
    if n_clicks is None or current_violation is None:
        raise PreventUpdate

    # record original prediction stats
    original_categories = current_violation.get("categories", [])
    original_severity = current_violation.get("severity", "medium")

    selected_categories = []
    for i, toggle_value in enumerate(toggle_values):
        if toggle_value:
            selected_categories.append(toggle_ids[i]["index"])

    updated_violation = current_violation.copy()
    updated_violation["categories"] = selected_categories
    updated_violation["severity"] = severity if severity else "medium"
    updated_violation["explanation"] = explanation
    updated_violation["verified"] = True

    # init toast
    show_toast = False
    toast_message = ""
    toast_header = ""
    toast_icon = "bi-bookmark-check"
    toast_color = "success"
    new_review_id = review_id

    # record verification stats to database
    try:
        db_manager = DatabaseManager()
        ai_agent = AI_Agent()
        first_save = False

        if not review_id:
            first_save = True

            unique_page_ids = set()
            for result in ai_results:
                page_id = result.get("location")
                if page_id not in unique_page_ids:
                    unique_page_ids.add(page_id)

            pages_scanned = len(unique_page_ids)
            total_violations = len(ai_results)

            new_review_id = db_manager.record_violation_review(
                pages_scanned=pages_scanned,
                total_violations=total_violations,
                status_code="completed",
                model=ai_agent.selected_model
            )
            print(f">>> AI Review ID created: {new_review_id}")

            db_manager.record_ai_analysis(new_review_id, ai_results)

            # update toast
            show_toast = True
            toast_message = f"{len(unique_page_ids)} pages and {len(ai_results)} violations have been saved."
            toast_header = "AI Review Saved to Database"

            print(f">>> Created new review entry: {review_id}")
        else:
            new_review_id = review_id
            print(f">>> Using existing review ID: {new_review_id}")

        db_manager.record_user_ai_verification_stats(
            review_id=new_review_id,
            original_categories=original_categories,
            modified_categories=selected_categories,
            original_severity=original_severity,
            modified_severity=severity if severity else "medium",
        )

        print(f">>> User Verification Stats recorded with AI Review ID: {review_id}")
    except Exception as e:
        print(f">>> Error recording AI User Verification stats: {e}")
        if first_save:
            show_toast = True
            toast_message = f"Error saving to database: {str(e)}"
            toast_header = "Error Saving Review"
            toast_icon = "bi-bookmark-dash"
            toast_color = "danger"

    updated_ai_results = []
    for result in ai_results:
        if result.get("text") == current_violation.get("text"):
            updated_ai_results.append(updated_violation)
        else:
            updated_ai_results.append(result)

    # Recreate the display with updated data
    encryption_manager = EncryptionManager()
    api_key = encryption_manager.decrypt_data(encrypted_api_key).decode()
    analysis_cards = []

    for violation in updated_ai_results:
        card = create_display(violation, api_key)
        analysis_cards.append(card)

    return updated_ai_results, analysis_cards, show_toast, toast_message, toast_header, toast_icon, toast_color, new_review_id


@callback(
    [
        Output("ai-analysis-results-store", "data", allow_duplicate=True),
        Output("ai-agent-page-content", "children", allow_duplicate=True),
        Output("review-id-store", "data", allow_duplicate=True),
    ],
    Input("refresh-api-button", "n_clicks"),
    State("verified-api-key-store", "data"),
    prevent_initial_call=True
)
def refresh_api_data(n_clicks, encrypted_api_key):
    """Force refresh data from API when button is clicked"""
    if n_clicks is None:
        raise PreventUpdate

    ai_agent = AI_Agent()
    encryption_manager = EncryptionManager()
    api_key = encryption_manager.decrypt_data(encrypted_api_key).decode()
    notion_manager = NotionManager(api_key=api_key)

    all_text_string, json_results = notion_manager.return_all_extraction_results()

    # get ai analysis
    ai_analysis = ai_agent.analyze_notion_using_gemini(json_results)
    ai_analysis_json = json.loads(ai_analysis)
    analysis_cards = []

    if not ai_analysis_json:
        return [], [dbc.Alert(
            "No violations found",
            color="success",
        )]

    for violation in ai_analysis_json:
        analysis_cards.append(create_display(violation, api_key))

    return ai_analysis_json, analysis_cards, None

# Remediation modals
@callback(
    [
        Output("remediation-modal", "is_open"),
        Output("remediation-display", "children"),
        Output("ai-suggestion", "children"),
        Output("remediation-suggestions-store", "data"),
    ],
    [
        Input({"type": "remediation-button", "index": ALL}, "n_clicks"),
    ],
    [
        State("remediation-modal", "is_open"),
        State("ai-analysis-results-store", "data"),
        State("remediation-results-store", "data"),
    ],
    prevent_initial_call=True

)
def toggle_remediation_modal(remediation_button, is_open, ai_results, remediation_results):
    """ Open and close modal for remediation button, handle AI suggestions and remediation options"""
    ctx_triggered = ctx.triggered_id
    #print(f"Button clicked {ctx_triggered}")

    if ctx_triggered is None:
        raise PreventUpdate

    if isinstance(ctx_triggered, dict) and ctx_triggered.get("type") == "remediation-button":
        if not any(click and click > 0 for click in remediation_button):
            raise PreventUpdate

        idx = ctx_triggered["index"]

        violation_data = None
        for result in ai_results:
            if result.get("text", "")[:20] == idx:
                violation_data = result
                break

        if not violation_data:
            raise PreventUpdate

        display_content = [
            html.P(f"Violation Text: {violation_data.get('text', 'No text')}"),
            html.P(f"Severity: {violation_data.get('severity', 'medium')}"),
            html.P(f"Categories: {', '.join(violation_data.get('categories', []))}"),
            html.P(f"Location: {violation_data.get('location', 'Unknown')}")
        ]

        # get remediation suggestions
        try:
            ai_agent = AI_Agent()

            ai_query = {
                "text": violation_data.get("text", ""),
                "categories" : violation_data.get("categories", []),
                "severity": violation_data.get("severity", "medium"),
                "location": violation_data.get("location", "Unknown")
            }

            suggestions = ai_agent.suggest_remediation_actions([ai_query])


            if suggestions and len(suggestions) > 0:
                suggestion = suggestions[0]
                # print(">>> Suggestion details:", suggestion)
                # print(">>> Violation location:", violation_data.get("location"))
                # print(">>> Sensitive text:", suggestion.get("sensitive_text"))
                return True, display_content, html.Div([
                    html.H4("AI Suggested Replacement:"),
                    html.P(suggestion.get("replacement_text", "No replacement suggested"))
                ]), suggestions
        except Exception as e:
            print(f">>> Error trying to retrieve AI suggestion for remediation: {e}")


        return True, display_content, "No suggestions available", None

    return False, [], [], None

#apply remediation when button is clicked
@callback(
    [
        Output("remediation-status", "children",allow_duplicate=True),
        Output("remediation-toast", "is_open"),
        Output("remediation-toast", "children"),
        Output("remediation-toast", "header"),
        Output("remediation-results-store", "data"),
        Output("review-id-store", "data", allow_duplicate=True),
    ],
    Input("apply-remediation-button", "n_clicks"),
    [
        State("remediation-suggestions-store", "data"),
        State("verified-api-key-store", "data"),
        State("review-id-store", "data")
    ],
    prevent_initial_call=True
)
def apply_remediation_button(n_clicks, remediation_suggestions, verified_api_key, review_id):
    """Apply remediation suggestions from AI to Notion when button is clicked"""
    if n_clicks is None or not remediation_suggestions:
        raise PreventUpdate

    try:
        ai_agent = AI_Agent()
        encryption_manager = EncryptionManager()
        api_key = encryption_manager.decrypt_data(verified_api_key).decode()
        notion_manager = NotionManager(api_key=api_key)
        db_manager = DatabaseManager()

        results = ai_agent.apply_ai_remediation_suggestions(remediation_suggestions, notion_manager=notion_manager)

        status_display = []
        status_text = ""

        if results.get("details"):
            status_display.append(html.H4("Details:"))
            for i, detail in enumerate(results['details']):
                status = detail.get("status", "unknown")
                color = "success" if status == "completed" else "danger"
                status_text = html.H3("Success!") if status == "completed" else html.H3("Error!")

                detail_content = [
                    html.P(f"Remediation {i + 1}: {status.upper()}", className="fw-bold"),
                    html.P(f"Sensitive text: {detail.get('sensitive_text', 'N/A Error')}"),
                    html.P(f"Replacement: {detail.get('replacement_text', 'N/A Error')}")
                ]

                if detail.get("error"):
                    detail_content.append(html.P(f"Remediation {i + 1}: {detail.get('error')}", className="text-danger"))

                status_display.append(dbc.Alert(detail_content, color=color, className="mt-2 mb-3"))
                status_display.insert(0,status_text)

        if not review_id:
            unique_page_ids = set()
            for suggestion in remediation_suggestions:
                if 'violation' in suggestion and 'location' in suggestion['violation']:
                    page_id = suggestion['violation']['location']
                    if page_id and page_id not in unique_page_ids:
                        unique_page_ids.add(page_id)

            review_id = db_manager.record_violation_review(
                pages_scanned=len(unique_page_ids),
                total_violations=len(remediation_suggestions),
                status_code="completed",
                model=ai_agent.selected_model
            )
            print(f">>> Created new review ID for AI remediation: {review_id}")

        if review_id and results and results.get("details"):
            for detail in results.get("details"):
                page_id = detail.get("location", "")
                page_url = notion_manager.get_page_url(page_id)
                block_ids = [detail.get("block_id", "")] if detail.get("block_id") else []

                db_manager.record_ai_remediation(
                    review_id=review_id,
                    page_id=page_id,
                    page_url=page_url,
                    original_text=detail.get("sensitive_text", ""),
                    masked_text=detail.get("replacement_text", ""),
                    block_ids=block_ids,
                    applied_to_notion_success=(detail.get("status") == "completed")
                    )
        return [
            status_display,
            True,
            f"Applied {results['successful_remediations']} of {results['total_suggested_remediations']} remediation suggestions",
            "Remediation Results",
            results,
            review_id]

    except Exception as e:
        print(f">>> Unable to apply remediation suggestions: {e}")
        error_display = [
            html.Div([
                html.H4("Error Applying Remediation:", className="text-danger"),
                html.P(str(e))
            ])
        ]
        return [
            error_display,
            True,
            f"Error applying remediation: {str(e)}",
            "Remediation Failed",
            None,
            review_id]

@callback(
    Output("remediation-status", "children"),
    [
        Input({"type": "remediation-button", "index": ALL}, "n_clicks"),
    ],
    prevent_initial_call=True
)
def clear_remediation_status(n_clicks):
    """Clear the remediation store when opening a new remediation modal"""
    return []



