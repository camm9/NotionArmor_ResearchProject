import requests
from notion_client import Client
from dotenv import load_dotenv
import os, json


class NotionManager:
    load_dotenv()

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("NOTION_API_KEY")
        self.client = Client(auth=api_key)
        if not self.api_key:
            print(">>> Error: No Notion API key set.")

    def notion_connection(self):
        """Make the connection to Notion API"""
        try:
            # Returns JSON response of databases and workspaces
            response = self.client.search()
            print(">>> Connection to Notion API successful")
            # print(json.dumps(response, indent=2))
            return response
        except Exception as e:
            print(f">>> Connection to Notion API failed: {e}")
            raise

    def test_notion_connection(self, api_key):
        """ Test connection to Notion API is functioning correctly"""
        try:
            headers = {
                "AUTHORIZATION": f"Bearer {api_key}",
                "Notion-Version": "2022-02-22"
            }
            response = requests.get("https://api.notion.com/v1/users/me", headers=headers)

            if response.status_code == 200:
                print(f">>> Connection to Notion API successful: {response.status_code}")
                return True
            else:
                print(f">>> Connection to Notion API unsuccessful: Status Code {response.status_code}")
                return False
        except Exception as e:
            print(f">>> Connection to Notion API failed: {e}")
            raise

    def print_extraction_results(self):
        api_response = self.notion_connection()
        dict_of_pages = self.get_page_list(api_response)

        for page_id in dict_of_pages:
            print(page_id + " - - - " + dict_of_pages[page_id]["title"])
            print(self.get_page_url(page_id))
            dictionary_plain_text, plain_text_string, dict_table_plain_text = self.get_page_content(page_id)
            print(dictionary_plain_text)
            print(dict_table_plain_text)
            print(plain_text_string)


    def return_all_extraction_results(self):
        api_response = self.notion_connection()
        dict_of_pages = self.get_page_list(api_response)
        dict_response = {}
        all_text = ""
        for page_id in dict_of_pages:
            dictionary_plain_text, plain_text_string, dict_table_plain_text = self.get_page_content(page_id)
            all_text += plain_text_string

            dict_response[page_id] = { "Plain Text ": plain_text_string,
                                       "Dictionary of Plain Text ": dict_table_plain_text,
                                       "Dictionary of Plain Text From Tables": dict_table_plain_text}

        return all_text, dict_response

    def extract_plain_text(self, dictionary_to_extract):
        texts = []
        for item in dictionary_to_extract:
            if isinstance(item, dict) and 'plain_text' in item:
                texts.append(item['plain_text'])
            elif isinstance(item, list):
                for toggle in item:
                    if isinstance(toggle, dict) and 'all_plain_text' in toggle:
                        texts.extend(toggle['all_plain_text'])
        return texts

    def return_notion_dict_of_pages(self):
        api_response = self.notion_connection()
        dict_of_pages = self.get_page_list(api_response)

        return dict_of_pages

    def get_page_url(self, page_id):
        """Find & Get the URL of the Notion page"""
        try:
            page = self.client.pages.retrieve(page_id)
            page_url = page["url"]
            print(">>> Retrieving page URL...")
            return page_url
        except Exception as e:
            print(f">>> Error retrieving page URL: {e}")
            raise

    def get_database_list(self, api_response):
        """Find & Get list of the Notion databases"""
        dict_of_databases = {}

        for item in api_response["results"]:
            if item["object"] == "database":
                database_id = item["id"]
                database_name = item["title"][0]["plain_text"]
                database_url = item["url"]

                dict_of_databases[database_id] = {
                    "name": database_name,
                    "url": database_url
                }

        print(">>> Databases found...")
        # print(json.dumps(dict_of_databases, indent=2))
        return dict_of_databases

    def get_page_list(self, api_response):
        dict_of_pages = {}

        for item in api_response["results"]:
            if item["object"] == "page":
                page_id = item["id"]

                try:
                    title = None
                    for key, value in item["properties"].items():
                        if value["type"] == "title":
                            title = value["title"][0]["text"]["content"]
                            break

                    dict_of_pages[page_id] = {
                        "id": page_id,
                        "title": title,
                    }

                except Exception as e:
                    print(f">>> A parsing error occurred: {e}")

        print(">>> pages found...")
        return dict_of_pages

    def get_page_content(self, page_id):
        """Find & Get the content of the Notion page
        Requirements: we need block object ID to get page content,
        blocks are separate to page properties
        """
        block_id = self.client.blocks.children.list(page_id)
        print(">>> Retrieving page blocks content... This may take a while...")
        plain_text_dictionary, plain_text_string, plain_text_table_dictionary = self.get_plain_text(block_id)
        return plain_text_dictionary, plain_text_string, plain_text_table_dictionary

    def get_plain_text(self, block_id):
        """Extract plain text from Notion JSON response
        Return a list of dictionaries saying where the text comes from
        and a simple text of all found plain text.
        """
        text_items = []
        plain_text_string = ""
        table_items = []
        toggle_items = []

        try:
            print(">>> Extracting plain text...")
            for block in block_id["results"]:
                block_type = block["type"]
                block_content = block.get(block_type, {})
                rich_text = block_content.get("rich_text", [])

                for item in rich_text:
                    if "plain_text" in item and item["plain_text"] != "":
                        # Create a string of all found plain_text
                        plain_text_string += item["plain_text"]
                        plain_text_string += "\n"
                        # Create list of dictionaries for each plain text and source
                        text_items.append({
                            "plain_text": item["plain_text"],
                            "block_type": block_type,
                            "block_id": block["id"],
                            "block_parent_id": block["parent"]["page_id"]
                        })

                if block_type == "table":
                    print(">>> Table found, extracting table plain text...")
                    # get table ID
                    table_id = block["id"]
                    # use table ID to get table rows from query to Retrieve block children
                    table_block = self.client.blocks.children.list(table_id)
                    table_properties = table_block["results"]

                    table_data = {
                        "table_id": table_id,
                        "block_parent_id": block["parent"]["page_id"],
                        "all_plain_text": []
                    }

                    for row in table_properties:
                        cells = row["table_row"]["cells"]
                        for cell in cells:
                            if cell != "" and len(cell) > 0:
                                cell_plain_text = cell[0].get("plain_text", "")
                                # Add to plain_text_string
                                plain_text_string += cell_plain_text
                                plain_text_string += "\n"
                                # Add to plain_text to table_data
                                table_data["all_plain_text"].append(cell_plain_text)
                            else:
                                # Handle for empty cell
                                continue
                    # Add single table info to table_items dictionary
                    table_items.append(table_data)

                if block_type == "toggle":
                    print(">>> Toggle found, extracting toggle plain text...")
                    plain_text_toggle = []
                    types_of_toggles = []
                    toggle_id = block["id"]
                    toggle_block = self.client.blocks.children.list(toggle_id)
                    toggle_properties = toggle_block["results"]
                    for value in toggle_properties:
                        toggle_type = value["type"]
                        toggle_content = value[toggle_type]
                        types_of_toggles.append(toggle_type)
                        if "rich_text" in toggle_content and toggle_content["rich_text"] != "":
                            toggle_rich_text = toggle_content["rich_text"]
                            for item in toggle_rich_text:
                                plain_text_toggle.append(item["plain_text"])
                                plain_text_string += item["plain_text"]
                                plain_text_string += "\n"

                    toggle_data = {
                        "all_plain_text": plain_text_toggle,
                        "toggle_id": toggle_id,
                        "types_of_toggles": types_of_toggles,
                        "block_parent_id": block["parent"]["page_id"],
                    }
                    # add toggle dictionary to text_items dictionary
                    toggle_items.append(toggle_data)
                    text_items.append(toggle_items)
                    print(">>> Toggle data extracted...")

        except Exception as e:
            print(">>> Error extracting plain text:...")
            raise

        if plain_text_string == "" and text_items == []:
            print(">>> No plain text found...")
        else:
            print(">>> Plain text extracted...")

        return text_items, plain_text_string, table_items

    def create_database_page(self, page_title, page_content, parent_page_id):
        """Create a new Notion page"""
        try:
            print(">>> Creating page...")
            page = self.client.pages.create(
                parent={"database_id": parent_page_id},
                properties={
                    "title": [{"text": {"content": page_title}}]
                },
                children=[
                    {
                        "object": "block",
                        "type": "heading_1",
                        "heading_1": {"rich_text": [{"type": "text", "text": {"content": page_content}}]}
                    }
                ]
            )
        except Exception as e:
            print(f">>> Error creating page: {e}")
            raise

    def get_list_of_users_and_bots(self):

        """Find & Get list of the Notion users & bots for workspace"""
        dict_of_users = {}
        dict_of_bots = {}
        try:
            list = self.client.users.list()
            print(
                ">>> Retrieving list of users & bots for workspace..."
            )
            for item in list["results"]:
                user_type = item["type"]
                user_id = item["id"]
                user_name = item["name"]
                user_email = item.get("person", {}).get("email", None)
                bot_workspace = item.get("bot", {}).get("workspace_name", None)

                if user_type == "bot":
                    dict_of_bots[user_id] = {
                        "bot_name": user_name,
                        "workspace_name": bot_workspace
                    }
                else:
                    dict_of_users[user_id] = {
                        "name": user_name,
                        "email": user_email
                    }

            print(">>> Users found...")

            return dict_of_users, dict_of_bots

        except json.JSONDecodeError as e:
            print(f">>> Error parsing JSON: {e}")
        except Exception as e:
            print(f">>> Unable to retreive users: {e}")
            raise

    def update_block_with_redacted_text(self, block_id, block_type, sensitive_text, redacted_text):
        """" Take user's input for text to be redact from review_violations and update notion workspace
        with masked sensitive text
        """
        success = False
        block = self.client.blocks.retrieve(block_id=block_id)
        try:
            if block_type == "paragraph":
                rich_text = block["paragraph"]["rich_text"]
                self._update_rich_text(block_id, "paragraph", rich_text, sensitive_text, redacted_text)

            elif block_type == "heading_1":
                rich_text = block["heading_1"]["rich_text"]
                self._update_rich_text(block_id, "heading_1", rich_text, sensitive_text, redacted_text)

            elif block_type == "heading_2":
                rich_text = block["heading_2"]["rich_text"]
                self._update_rich_text(block_id, "heading_2", rich_text, sensitive_text, redacted_text)

            elif block_type == "heading_3":
                rich_text = block["heading_3"]["rich_text"]
                self._update_rich_text(block_id, "heading_3", rich_text, sensitive_text, redacted_text)

            elif block_type == "bulleted_list_item":
                rich_text = block["bulleted_list_item"]["rich_text"]
                self._update_rich_text(block_id, "bulleted_list_item", rich_text, sensitive_text, redacted_text)

            elif block_type == "numbered_list_item":
                rich_text = block["numbered_list_item"]["rich_text"]
                self._update_rich_text(block_id, "numbered_list_item", rich_text, sensitive_text, redacted_text)

            elif block_type == "quote":
                rich_text = block["quote"]["rich_text"]
                self._update_rich_text(block_id, "quote", rich_text, sensitive_text, redacted_text)

            elif block_type == "to_do":
                rich_text = block["to_do"]["rich_text"]
                self._update_rich_text(block_id, "to_do", rich_text, sensitive_text, redacted_text)

            elif block_type == "toggle":
                # parent block
                rich_text = block["toggle"]["rich_text"]
                self._update_rich_text(block_id, "toggle", rich_text, sensitive_text, redacted_text)

                # issue with toggle children
                toggle_block = self.client.blocks.children.list(block_id)
                toggle_properties = toggle_block["results"]

                for value in toggle_properties:
                    toggle_child_type = value["type"]
                    toggle_child_id = value["id"]
                    if toggle_child_type in ["paragraph", "heading_1", "heading_2", "heading_3",
                                      "bulleted_list_item", "numbered_list_item", "quote", "to_do"]:
                        toggle_child_rich_text = value[toggle_child_type]["rich_text"]
                        self._update_rich_text(toggle_child_id, toggle_child_type, toggle_child_rich_text, sensitive_text, redacted_text)


            elif block_type == "table":
                # tables require special handling, we need to fetch and update each row, block id = table id
                table_rows = self.client.blocks.children.list(block_id)
                for row in table_rows["results"]:
                    if row["type"] == "table_row":
                        row_id = row["id"]
                        cells = row["table_row"]["cells"]
                        updated_cells = []
                        for cell in cells:
                            updated_cell = []
                            for text_item in cell:
                                if "plain_text" in text_item and sensitive_text in text_item["plain_text"]:
                                    # new text item with redacted content
                                    new_text = text_item["plain_text"].replace(sensitive_text, redacted_text)
                                    updated_cell.append({
                                        "type": "text",
                                        "text": {
                                            "content": new_text,
                                            "link": text_item.get("text", {}).get("link")
                                        }
                                    })
                                else:
                                    updated_cell.append(text_item)
                            updated_cells.append(updated_cell)

                        # update the row with redacted cells
                        self.client.blocks.update(
                            block_id=row_id,
                            table_row={
                                "cells": updated_cells
                            }
                        )
            success = True

        except Exception as e:
            print(f">>> Error updating block: {e}")
            success = False
            return success
        if success:
            print(">>> Redactions successfully applied to Notion workspace")
        else:
            print(">>> Redactions failed to be applied to Notion workspace")
        return success

    def _update_rich_text(self, block_id, block_type, rich_text, sensitive_text, redacted_text):
        """ Organize the rich text to be updated with Notion API call in update_block_with_redacted_text """
        success = None
        try:
            # keep track of rich text
            updated_rich_text = []
            for text_item in rich_text:
                # verify sensitive text is in plain_text
                if "plain_text" in text_item and sensitive_text in text_item["plain_text"]:
                    new_text = text_item["plain_text"].replace(sensitive_text, redacted_text)
                    updated_rich_text.append({
                        "type": "text",
                        "text": {
                            "content": new_text,
                            "link": text_item.get("text", {}).get("link")
                        },
                        "annotations": text_item.get("annotations", {})
                    })
                else:
                    updated_rich_text.append(text_item)

            # update the block with redacted rich_text
            update_data = {
                block_type: {
                    "rich_text": updated_rich_text
                }
            }
            self.client.blocks.update(
                block_id=block_id,
                **update_data
            )
            success = True
            return success
        except Exception as e:
            print(f">>> Error updating rich text: {e}")
            success = False
            return success

    def _find_block_with_text(self, page_id, text_to_find):
        """ Find the block with the given text and given page id
        NOTE: IN NOTION PAGE IDS AND BLOCK IDS ARE DIFFERENT """

        # get page content
        plain_text_dictionary, plain_text_string, plain_text_table_dictionary = self.get_page_content(page_id)

        for item in plain_text_dictionary:
            #handle most block types
            if isinstance(item, dict) and "plain_text" in item:
                if text_to_find in item.get("plain_text", ""):
                    return {
                        "block_id": item["block_id"],
                        "block_type": item["block_type"]
                    }
            #handle toggle values
            elif isinstance(item, list):
                for toggle in item:
                    if isinstance(toggle, dict) and toggle["all_plain_text"]:
                        for plain_text in toggle.get("all_plain_text", []):
                            if isinstance(plain_text, str) and text_to_find in plain_text:
                                return {
                                    "block_id": toggle["toggle_id"],
                                    "block_type": "toggle"
                                }
        # check tables
        for table in plain_text_table_dictionary:
            if "all_plain_text" in table:
                for cell_text in table["all_plain_text"]:
                    if text_to_find in cell_text:
                        return {
                            "block_id": table["table_id"],
                            "block_type": "table"
                        }

        return None # no text found in blocks

    def get_page_title(self, page_id):
        """ Get the page title of the given page id """
        try:
            page = self.client.pages.retrieve(page_id)
            for prop_id, prop_data in page.get("properties", {}).items():
                if prop_data.get("type") == "title":
                    title_items = prop_data.get("title", [])
                    if title_items:
                        return title_items[0].get("plain_text", "")

            # If no title property found
            return f"Untitled Page ({page_id[:8]}...)"
        except Exception as e:
            print(f">>> Error retrieving page title: {e}")
            return f"Page {page_id[:8]}..."
