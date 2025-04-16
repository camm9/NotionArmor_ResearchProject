import dash
from encryption_manager import EncryptionManager
from database_manager import DatabaseManager
from dash.exceptions import PreventUpdate
from dash import html, dcc, callback, Input, Output, ctx, ALL, no_update, State
import dash_bootstrap_components as dbc
from notion_client_manager import NotionManager
from compliance_engine import ComplianceEngine
from datetime import datetime

""" Home page in Dash for Front End """

dash.register_page(__name__)


def get_notion_data(encrypted_api_key):
    """Get Notion Data"""
    try:
        encryption_manager = EncryptionManager()
        # turn the stored string back into bytes for decryption
        api_key = encryption_manager.decrypt_data(encrypted_api_key).decode()
        notion_manager = NotionManager(api_key=api_key)
        # Get API results
        api_response = notion_manager.notion_connection()

        # Get pages and their content
        pages_dict = notion_manager.get_page_list(api_response)

        # Format data for display
        formatted_data = []
        for page_id, page_info in pages_dict.items():
            try:
                page_url = notion_manager.get_page_url(page_id)
                content_dict, content_text, table_data = notion_manager.get_page_content(page_id)

                page_data = {
                    "title": page_info["title"],
                    "url": page_url,
                    "content": content_dict,
                    "tables": table_data,
                    "plain_text": content_text
                }
                formatted_data.append(page_data)

            except Exception as e:
                print(f">>> Error processing page {page_id}: {e}")
                continue

        return True, formatted_data
    except Exception as e:
        return False, f">>> Error connecting to Notion: {str(e)}"


def load_compliance_engine():
    """Load Vectorizer and Classifier"""
    compliance_engine = ComplianceEngine()
    soc2_classifier_loaded, soc2_vectorizer_loaded = compliance_engine.load_trained_model("SOC2_Model")
    return soc2_classifier_loaded, soc2_vectorizer_loaded


def init_compliance_review(soc2_classifier_loaded, soc2_vectorizer_loaded, notion_data):
    """Review Compliance of Notion Data"""
    compliance_engine = ComplianceEngine()
    violations = []

    for page in notion_data:
        all_text_on_page = []
        all_text_on_page.append(page["plain_text"])
        if page["tables"]:
            for table in page["tables"]:
                table_text = "\n".join(table["all_plain_text"])
                all_text_on_page.append(table_text)

        all_text_to_review = "\n".join(all_text_on_page)
        x = soc2_vectorizer_loaded.transform([all_text_to_review])
        prediction = soc2_classifier_loaded.predict(x)[0]

        current_violations = []
        for index, label in enumerate(compliance_engine.soc2_labels):
            if prediction[index] == 1:
                current_violations.append(label)

        violations.append({
            "page_title": page["title"],
            "page_url": page["url"],
            "text_to_review": all_text_to_review,
            "violations": current_violations
        })

    return violations


def save_violations_review_to_db(notion_data, model_predictions, redact_store=None):
    """ Save Violations Review to performance db"""
    try:
        db_manager = DatabaseManager()
        compliance_engine = ComplianceEngine()

        pages_scanned = len(notion_data)

        # remove compliant from violations count
        total_violations = 0
        for prediction in model_predictions:
            pred = prediction.get("violations", [])
            total_violations += len(pred)

        for prediction in model_predictions:
            if "compliant" in prediction.get("violations", []):
                total_violations -= 1

        model = compliance_engine.selected_model
        review_id = db_manager.record_violation_review(
            pages_scanned=pages_scanned, total_violations=total_violations, status_code="completed", model=model
        )

        # Record the initial model predictions
        success = db_manager.record_model_violations(review_id, notion_data, model_predictions)

        # save redaction data to db if redactions were made
        if redact_store:
            for page_title, redactions in redact_store.items():
                page_id = None
                for page in notion_data:
                    if page["title"] == page_title:
                        page_id = page["url"].split("-")[-1]
                        break

                if page_id:
                    for redaction in redactions:
                        block_ids = [location["block_id"] for location in redaction.get("text_locations", [])]

                        db_manager.record_user_redactions(
                            review_id,
                            page_id,
                            page_title,
                            redaction["original_text"],
                            redaction["masked_text"],
                            block_ids,
                            redaction.get("applied_to_notion", False)
                        )

        print(f">>> Violations Review saved to database with ID: {review_id}")
        return True, review_id, f"Violations Review saved to database with ID: {review_id}"

    except Exception as e:
        print(f">>> Error connecting/updating to performance database: {str(e)}")
        error_message = f">>> Error connecting/updating to performance database: {str(e)}"
        return False, None, error_message

def create_feedback_display():
    """ Create Feedback Display to Edit Findings"""
    return dbc.Modal([
        dbc.ModalHeader(html.H1(f"Verify Findings")),
        dbc.ModalBody([
            html.Div(id="current-violations-display"),
            # create toggles for model prediction labels
            html.P("Adjust the labels below: "),
            html.Div([

                dbc.Switch(
                    value=False,  # get value of original prediction
                    id={"type": "violation-switch", "index": label},  # make a match type dict
                    label=label.capitalize(),

                ) for label in ComplianceEngine().soc2_labels
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("Submit Feedback", id="submit-feedback", color="primary")
        ]),
    ],
        id="feedback-modal",
        is_open=False,
    )

def mask_text(text):
    """ Redact Text By Creating A Mask of * """
    return "*" * len(text)

def create_redact_display():
    """ Create Redaction Display to Redact Sensitive Findings"""
    return dbc.Modal([
        dbc.ModalHeader(html.H1(f"Redact Sensitive Data")),
        dbc.ModalBody([
            html.Div(id="redact-display"),
            # create toggles for model prediction labels
            html.P("Enter text to redact: "),
            html.Div([
                dbc.Input(id="redact-input", placeholder="Enter text to redact"),
            ]),
            html.Div(id="redact-preview")
        ]),
        dbc.ModalFooter([
            dbc.Button("Apply Redaction", id="apply-redaction", color="primary")
        ]),
    ],
        id="redact-modal",
        is_open=False,
    )


def create_display(notion_data, violations_data, redact_store=None):
    """Display Notion Data"""
    # check if user verified
    is_verified = False
    if isinstance(violations_data, dict):
        is_verified = violations_data.get("user_feedback", False)
        violations = violations_data.get("violations", [])
    else:
        violations = violations_data

    if "compliant" in violations or not violations:
        card_color = "lite"
    else:
        card_color = "warning"

    # check for redaction
    has_redactions = False
    page_title = notion_data["title"]

    if redact_store and page_title in redact_store and redact_store[page_title]:
        for redaction in redact_store[page_title]:
            if redaction.get("applied_to_notion", False):
                has_redactions = True
                break

    table_content = []
    if notion_data["tables"]:
        for table in notion_data["tables"]:
            table_content.append(html.H4(f"Table: {table['table_id']}"))
            for cell in table["all_plain_text"]:
                table_content.append(html.P(cell))
            table_content.append(html.Hr())

    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.H2(notion_data["title"], className="card-title"),
                    dbc.Badge("REMEDIATED",
                              color="purple",
                              pill=True,
                              className="ms-2") if has_redactions else None,
                    dbc.Badge("USER VERIFIED",
                              color="success",
                              pill=True,
                              className="ms-2") if is_verified else None], className="d-flex align-items-center"),
                    dbc.CardLink("View Source Page", href=notion_data["url"], target="_blank", className="card-link mb-3"),
            ]),
            dbc.CardBody([
                html.P(notion_data["plain_text"], className="card-text"),
                html.Div(table_content) if table_content else None,
                html.Div([
                    html.H4(f"Violations: {', '.join(violations)}") if violations else html.P("No Violations")
                ]),

            ]),
            dbc.Button(
                "Verify",
                id={"type": "decline-button", "index": notion_data["title"]},
                outline=True,
                color="danger",
                class_name="mx-auto mb-3",
                n_clicks=0
            ),
            dbc.Button(
                "Redact Text",
                id={"type": "redact-button", "index": notion_data["title"]},
                outline=True,
                color="danger",
                class_name="mx-auto mb-3",
                n_clicks=0
            ) if violations and "compliant" not in violations else None,

        ], color=card_color, className="mb-3"),
    ])


layout = html.Div([
    dcc.Location(id='url-location', refresh=True),
    dcc.Store(id="verified-api-key-store", storage_type="session"),
    dbc.Container(children=[
        html.H1('Violations Review'),
        dcc.Store(id="model-predictions-store"),
        dcc.Store(id="feedback-predictions-store"),
        dcc.Store(id="notion-data-store"),
        dcc.Store(id="current-page-store", data=None),  # store current page that is open in modal
        dcc.Store(id="redact-store", data={}),
        dcc.Store(id="review-id-store", data=None),
        create_feedback_display(),
        create_redact_display(),
        # Alert button triggered after feedback
        dbc.Offcanvas(
            html.P("Feedback submitted successfully!"),
            id="feedback-submitted",
            is_open=False,
            placement="bottom",
        ),
        dbc.Offcanvas(
            html.P("Model retrained successfully!"),
            id="retrain-submitted",
            is_open=False,
            placement="bottom",
        ),
        dcc.Interval(id="redirect-timer", interval=2000, max_intervals=1, disabled=True),
        dcc.Loading(
            id="loading-notion",
            children=[
                html.Div(id="notion-error"),
                html.Div(id="page-content"),
                dbc.Stack([
                    # Update Info Button
                    dbc.Button(
                        "Refresh Data",
                        id="ml-refresh-button",
                        color="info",
                        className=" mb-4 mx-auto"
                    ),


                    # Retrain Model with User Feedback
                    dbc.Button(
                        "Retrain Model with User Feedback",
                        id="retrain-button",
                        color="warning",
                        className=" mb-4 mx-auto"
                    ),
                    # Complete Review and Send User back to Dashboard
                    dbc.Button(
                        "Complete Review",
                        id="complete-review-button",
                        color="success",
                        className=" mb-4 mx-auto",
                        href="/dashboard"
                    ),
                ])
            ],
            type="cube",
            className="position-fixed top-50 start-50 translate-middle"
        ),
    ])
], className="m-4")


@callback(
    [
        Output("page-content", "children"),
        Output("model-predictions-store", "data", allow_duplicate=True),
        Output("notion-data-store", "data"),
        Output("review-id-store", "data", allow_duplicate=True),
    ],
    [Input("ml-refresh-button", "n_clicks"),
     Input("verified-api-key-store", "data"), ],
    [State("redact-store", "data")],
    prevent_initial_call='initial_duplicate'
)
def refresh_data_ml(n_clicks, api_key, redact_store):
    """Refresh Data"""

    if not api_key:
        return dbc.Alert(
            "No API Key Provided. Please return to home page and try again",
            color="danger",
        ), no_update, no_update
    success, data = get_notion_data(api_key)

    if not success:
        return dbc.Alert(
            data,
            color="danger",
        ), no_update, no_update

    soc2_classifier_loaded, soc2_vectorizer_loaded = load_compliance_engine()
    # out model_predictions to a store in layout
    model_predictions = init_compliance_review(soc2_classifier_loaded, soc2_vectorizer_loaded, data)

    # save data to database automatically when refreshing
    review_id = None

    if data and model_predictions:
        success, review_id, message = save_violations_review_to_db(data, model_predictions, redact_store)

    violation_cards = []
    for page_data, violation_data in zip(data, model_predictions):
        violation_cards.append(create_display(page_data, violation_data, redact_store))

    return violation_cards, model_predictions, data, review_id

@callback(
    [
        Output("feedback-modal", "is_open"),
        Output("current-violations-display", "children"),
        Output({"type": "violation-switch", "index": ALL}, "value"),
        Output("current-page-store", "data")  # store currently opened modal page
    ],
    [
        Input({"type": "decline-button", "index": ALL}, "n_clicks"),
        Input("submit-feedback", "n_clicks"),
        Input({"type": "violation-switch", "index": ALL}, "value")  # get toggle values regardless of state
    ],
    [
        State("feedback-modal", "is_open"),
        State("model-predictions-store", "data"),
        State({"type": "violation-switch", "index": ALL}, "id")
    ],
    prevent_initial_call=True
)
def open_feedback_modal(decline_clicks, submit_clicks, toggle_values, is_open, model_predictions, toggle_ids_state):
    """Open Feedback Modal"""
    # print(f"Modal callback triggered by: {ctx.triggered_id}")
    # print(f"Current modal state: {is_open}")

    if ctx.triggered_id is None or (
            isinstance(ctx.triggered_id, dict) and ctx.triggered_id.get("type") == "decline-button" and all(
        click in [0, None] for click in decline_clicks)):
        raise PreventUpdate

    if ctx.triggered_id == "submit-feedback":
        # print("Submit button clicked in open_feedback_modal.")
        empty_values = [False] * len(toggle_values)
        return False, [], empty_values, None

    instance_is_dict = isinstance(ctx.triggered_id, dict)
    check_dict_type = ctx.triggered_id.get("type") if instance_is_dict else None

    # if a toggle is changed we have to register the new state
    if instance_is_dict and check_dict_type == "violation-switch":
        toggle_states = {toggle_id["index"]: value for toggle_id, value in
                         zip(toggle_ids_state, toggle_values)}  # security : True

        if toggle_states.get("compliant", False):  # meaning compliance toggled on
            # set all other toggles to False
            new_toggle_values = [False] * len(toggle_values)
            new_toggle_values[toggle_ids_state.index({"type": "violation-switch", "index": "compliant"})] = True
            return no_update, no_update, new_toggle_values, no_update

        return is_open, no_update, toggle_values, no_update

    # if edit findings button clicked
    if instance_is_dict and check_dict_type == "decline-button":
        page_title = ctx.triggered_id["index"]
        page_violations = []
        for prediction in model_predictions:
            if prediction["page_title"] == page_title:
                page_violations = prediction["violations"]
                break

        # set toggle switch values if violations found
        labels = ComplianceEngine().soc2_labels
        toggle_values = [label.lower() in [v.lower() for v in page_violations] for label in labels]

        violations_found = ', '.join(page_violations) if page_violations else 'None'
        return True, [
            html.P(f"Page: {page_title}"),
            html.P(f"Violations Found: {violations_found}")
        ], toggle_values, page_title  # store page title in "current-page-store"

    # if the decline button is NOT clicked
    return is_open, no_update, [no_update] * len(toggle_ids_state), no_update


@callback(
    [  # outputs
        Output("model-predictions-store", "data", allow_duplicate=True),
        Output("feedback-submitted", "is_open", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
        Output("review-id-store", "data", allow_duplicate=True),
    ],
    [  # inputs
        Input("submit-feedback", "n_clicks"),
    ],
    [  # states
        State({"type": "violation-switch", "index": ALL}, "value"),  # True or False -> 1 or 0 for model
        State({"type": "violation-switch", "index": ALL}, "id"),  # label
        State("current-page-store", "data"),
        State("model-predictions-store", "data"),
        State("notion-data-store", "data"),
        State("review-id-store", "data"),
        State("redact-store", "data")
    ],
    prevent_initial_call=True
)
def handle_feedback(n_clicks, toggle_values, toggle_ids, current_page, model_predictions, notion_data, review_id, redact_store):
    """Handle user feedback when they edit the model predictions"""

    if not ctx.triggered_id or not current_page:
        raise PreventUpdate

    toggle_states = {  # { 'security' : False}
        toggle_id["index"]: value
        for toggle_id, value in zip(toggle_ids, toggle_values)
    }


    labels = ComplianceEngine().soc2_labels
    new_violations = [
        label
        for label in labels
        if toggle_states.get(label.lower(), False)
    ]

    # record original predictions
    original_violations = []
    for prediction in model_predictions:
        if prediction["page_title"] == current_page:
            original_violations = prediction.get("violations", [])
            break

    compliance_engine = ComplianceEngine()

    # record verification stats to db
    try:
        db_manager = DatabaseManager()

        model = compliance_engine.selected_model

        if not review_id:
            unique_page_ids = set()
            for data in notion_data:
                url = data.get("url","")
                if url and url not in unique_page_ids:
                    unique_page_ids.add(url)

            pages_scanned = len(unique_page_ids)
            total_violations = sum(len(pred.get("violations", [])) for pred in model_predictions)

            review_id = db_manager.record_violation_review(
                pages_scanned=pages_scanned,
                total_violations=total_violations,
                status_code="completed",
                model=compliance_engine.selected_model,
            )

            # record verification stats
            db_manager.record_user_model_verification_stats(
                review_id=review_id,
                original_categories=original_violations,
                modified_categories=new_violations,
            )

            # record redactions if existing
            if redact_store:
                success, _,_ = save_violations_review_to_db(notion_data, model_predictions, redact_store)

        # Record verification stats
        db_manager.record_user_model_verification_stats(
            review_id=review_id,
            original_categories=original_violations,
            modified_categories=new_violations,
        )
        print(f">>> User Verification Stats recorded for {model} with Review ID: {review_id} ")

    except Exception as e:
        print(f">>> Unable to record user verification/redaction stats to database: {e}")

    updated_model_predictions = []
    for prediction in model_predictions:
        if prediction["page_title"] == current_page:
            updated_prediction = {
                **prediction,  # keep original dictionary
                "violations": new_violations,
                "user_feedback": True
            }
            updated_model_predictions.append(updated_prediction)
        else:
            updated_model_predictions.append(prediction)

    violation_cards = []
    for page_data, violation_data in zip(notion_data, updated_model_predictions):
        violation_cards.append(create_display(page_data, violation_data, redact_store))

    return updated_model_predictions, True, violation_cards, review_id


@callback(
    [
        Output("model-predictions-store", "data", allow_duplicate=True),
        Output("retrain-submitted", "is_open"),
        Output("redirect-timer", "disabled"),
    ],
    Input("retrain-button", "n_clicks"),
    [
        State("model-predictions-store", "data"),
    ],
    prevent_initial_call=True
)
def retrain_model(n_clicks, model_predictions):
    """Retrain Model with User Feedback"""
    if n_clicks is None:
        raise PreventUpdate

    print(">>> Starting model retraining...")

    # Let's retrain the model with data that was corrected by the user
    feedback_data = {}
    for prediction in model_predictions:
        if prediction.get("user_feedback", False):
            text_to_review = prediction["text_to_review"]
            violations = prediction["violations"]
            feedback_data[text_to_review] = violations

    # if no user feedback, then don't train model
    if not feedback_data:
        print(">>> No user feedback found. Skipping model retraining...")
        return no_update, True, False

    # initiate classifier and vector, retrain model
    try:
        compliance_engine = ComplianceEngine()
        soc2_classifier_loaded, soc2_vectorizer_loaded = compliance_engine.load_trained_model("SOC2_Model")

        # don't do ask_user_retrain, skip to saving the data
        print(">>> Saving feedback data to training data...")
        compliance_engine.save_feedback_for_training(model_predictions, feedback_data)

        retrained_classifier, retrained_vectorizer = compliance_engine.retrain_model_with_new_feedback(
            soc2_classifier_loaded, soc2_vectorizer_loaded)

        print(">>> Model retraining complete!")
        return no_update, True, False

    except Exception as e:
        print(f">>> ERROR while retraining model: {e}")
        return no_update, True, True


@callback(
    [
        Output("url-location", "pathname"),
    ],
    [
        Input("redirect-timer", "n_intervals"),
    ],
    prevent_initial_call=True,
)
def redirect_to_dashboard(n_intervals):
    """Redirect to Dashboard"""

    if n_intervals is None:
        raise PreventUpdate
    print(">>> Redirecting to dashboard...")
    return ["/dashboard"]

@callback(
    [Output("redact-modal", "is_open"),
     Output("current-page-store", "data", allow_duplicate=True)
    ],
    Input({"type": "redact-button", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def open_redact_modal(n_clicks):
    """Open Redact Modal"""
    triggered_id = ctx.triggered_id

    if not triggered_id or not any(click and click > 0 for click in n_clicks):
        raise PreventUpdate

    if isinstance(triggered_id, dict) and triggered_id.get("type") == "redact-button":
        page_title = triggered_id["index"]
        return [True, page_title]

    return [False, no_update]

@callback(
    Output("redact-preview", "children"),
    Input("redact-input", "value")
)
def preview_redaction(sensitive_data):
    """Preview Redaction"""
    if not sensitive_data:
        return html.P("Enter text to see a preview of redaction.")

    redacted_text = mask_text(sensitive_data)

    return [
        html.P("Preview:"),
        dbc.Card([
            dbc.CardBody([
                html.P(f"Original: {sensitive_data}"),
                html.P(f"Redacted: {redacted_text}"),
            ])
        ])
    ]


@callback(
    [
        Output("notion-data-store", "data", allow_duplicate=True),
        Output("model-predictions-store", "data", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
        Output("redact-store", "data"),
        Output("redact-modal", "is_open", allow_duplicate=True),
    ],
    Input("apply-redaction", "n_clicks"),
    [
        State("redact-input", "value"),
        State("current-page-store", "data"),
        State("notion-data-store", "data"),
        State("model-predictions-store", "data"),
        State("redact-store", "data"),
        State("verified-api-key-store", "data"),
        State("review-id-store", "data")
    ],
    prevent_initial_call=True,
)
def apply_redaction(n_clicks, sensitive_data_to_redact, current_page, notion_data, model_predictions, redact_store,
                    encrypted_api_key, review_id):
    """Apply Redaction to Notion Workspace"""
    if not n_clicks or not sensitive_data_to_redact or not current_page:
        raise PreventUpdate

    # create store copies for Dash
    updated_notion_data = notion_data.copy()
    updated_model_predictions = model_predictions.copy()
    updated_redact_store = redact_store.copy() if redact_store else {}

    # get correct page data for notion
    page_index = None
    page_id = None
    for i, page in enumerate(updated_notion_data):
        if page["title"] == current_page:
            page_index = i
            # Extract page ID from URL
            page_id = page["url"].split("-")[-1]
            break

    if page_index is None:
        print(f">>> Error: Page {current_page} not found.")
        raise PreventUpdate

    # generate masked text
    redacted_text = mask_text(sensitive_data_to_redact)

    # Initialize redact_store for this page if needed
    if current_page not in updated_redact_store:
        updated_redact_store[current_page] = []

    # Notion api ids for text location
    text_locations = []

    # verify that text is in page
    text_found = False
    page_data = updated_notion_data[page_index]

    if sensitive_data_to_redact in page_data["plain_text"]:
        text_found = True
        updated_notion_data[page_index]["plain_text"] = page_data["plain_text"].replace(
            sensitive_data_to_redact, redacted_text
        )

        # Find which blocks contain the sensitive text
        for text_item in page_data["content"]:
            if isinstance(text_item, dict) and "plain_text" in text_item:
                if sensitive_data_to_redact in text_item["plain_text"]:
                    text_locations.append({
                        "block_id": text_item["block_id"],
                        "block_type": text_item["block_type"]
                    })

        # table data.....
        if page_data["tables"]:
            for j, table in enumerate(page_data["tables"]):
                for k, cell in enumerate(table["all_plain_text"]):
                    if sensitive_data_to_redact in cell:
                        text_found = True
                        updated_notion_data[page_index]["tables"][j]["all_plain_text"][k] = cell.replace(
                            sensitive_data_to_redact, redacted_text
                        )
                        text_locations.append({
                            "block_id": table["table_id"],
                            "block_type": "table"
                        })

    # If text wasn't found, alert the user
    if not text_found:
        print(f">>> Text '{sensitive_data_to_redact}' not found in the page")
        # Create some kind of alert or toast
        # in front end
        raise PreventUpdate

    for i, prediction in enumerate(updated_model_predictions):
        if prediction["page_title"] == current_page:
            if sensitive_data_to_redact in prediction["text_to_review"]:
                updated_model_predictions[i]["text_to_review"] = prediction["text_to_review"].replace(
                    sensitive_data_to_redact, redacted_text
                )


    updated_redact_store[current_page].append({
        "original_text": sensitive_data_to_redact,
        "masked_text": redacted_text,
        "timestamp": datetime.now().isoformat(),
        "text_locations": text_locations,
        "applied_to_notion": False
    })


    redaction_applied_to_notion = False
    notion_success = False
    db_success = False

    # Apply redaction to Notion workspace
    try:
        # Decrypt API key
        encryption_manager = EncryptionManager()
        api_key = encryption_manager.decrypt_data(encrypted_api_key).decode()
        notion_manager = NotionManager(api_key=api_key)


        for location in text_locations:
            success = notion_manager.update_block_with_redacted_text(
                location["block_id"],
                location["block_type"],
                sensitive_data_to_redact,
                redacted_text
            )
            if success:
                print(f">>> Successfully updated Notion content with redacted text in {location['block_type']}")
                updated_redact_store[current_page][-1]["applied_to_notion"] = True
                redaction_applied_to_notion = True
                notion_success = True

        # Save to database
        try:
            db_manager = DatabaseManager()
            compliance_engine = ComplianceEngine()

            # Create a new review if needed
            if not review_id:
                pages_scanned = len(notion_data)
                total_violations = sum(len(pred.get("violations", [])) for pred in model_predictions)
                for prediction in model_predictions:
                    if "compliant" in prediction.get("violations", []):
                        total_violations -= 1

                model = compliance_engine.selected_model
                review_id = db_manager.record_violation_review(
                    pages_scanned=pages_scanned,
                    total_violations=total_violations,
                    status_code="completed",
                    model=model
                )

            if page_id:
                block_ids = [location["block_id"] for location in text_locations]
                db_manager.record_user_redactions(
                    review_id,
                    page_id,
                    current_page,
                    sensitive_data_to_redact,
                    redacted_text,
                    block_ids,
                    redaction_applied_to_notion
                )
                db_success = True
                print(f">>> Redaction saved to database for review ID: {review_id}")

        except Exception as e:
            print(f">>> Error saving redaction to database: {e}")


    except Exception as e:
        print(f">>> Error applying redaction to Notion: {e}")


    violation_cards = []
    for page_data, violation_data in zip(updated_notion_data, updated_model_predictions):
        violation_cards.append(create_display(page_data, violation_data,updated_redact_store))

    return (
        updated_notion_data,
        updated_model_predictions,
        violation_cards,
        updated_redact_store,
        False,
    )