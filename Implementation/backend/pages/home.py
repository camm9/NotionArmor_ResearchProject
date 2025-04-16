import dash
from encryption_manager import EncryptionManager
from dash.exceptions import PreventUpdate
from dash import html, dcc, callback, Input, Output, ctx, ALL, no_update, State
import dash_bootstrap_components as dbc
from notion_client_manager import NotionManager
from compliance_engine import ComplianceEngine
from ai_agent import AI_Agent


""" Home page in Dash for Front End """

dash.register_page(__name__, path='/')

def create_model_dropdown():
    model_list = []
    dropdown_list = []

    compliance_engine = ComplianceEngine()
    ai_agent = AI_Agent()
    for model in compliance_engine.get_selected_models():
        model_list.append(model)

    for model in ai_agent.get_selected_models():
        model_list.append(model)

    #create a short name for dash id and add to drop down
    for model in model_list:
        short_name = model.split(' ')[0]

        if '-' in short_name:
            short_name = short_name.split('-')[0]

        dropdown = dbc.DropdownMenuItem(model, id=f"{short_name}-menu-item", n_clicks=0)
        dropdown_list.append(dropdown)

    return dropdown_list

layout = html.Div([
    dcc.Location(id='url-location', refresh=True),
    dcc.Interval(id="redirect-violations-timer", interval=2000, max_intervals=1, disabled=True),
    html.Div([
        html.H2("Please enter your Notion API key and enable internal integration in pages you wish to investigate.", className='text-center'),
        dbc.Input(id="input-api-key", autofocus=True, size="lg", type="text", className="mb-3",
                  placeholder="Enter your Notion API KEY.....", maxLength=50, minLength=50),
        dcc.Store(id="verified-api-key-store", storage_type="session"),
        dcc.Store(id="model-selected-store"),
        dbc.Alert(
            [
                html.H1("Error"),
                html.P(id="error-message"),
            ],
            id="input-api-key-alert",
            color="warning",
            dismissable=True,
            fade=True,
            is_open=False,
            className="text-center mb-3",
        ),
        dbc.Toast(
            "API Verified!",
            id="success-toast",
            header="Success!",
            is_open=False,
            color="success",
            duration=1500,
            className="position-fixed top-50 start-50 translate-middle text-center mb-3"
        ),
        dbc.Button("Verify API", id="submit-api-key-button", n_clicks=0, color="primary", className="d-block mx-auto mb-3"),
        html.Div([
            dbc.DropdownMenu(
            label="Model Selection",
            color="secondary",
            size="lg",
            children=
                create_model_dropdown(),
            )
        ], className="d-flex justify-content-center mb-3"),
        html.Div([
            html.P(id="model-selected-text", className="text-center mb-3"),
        ]),
        dbc.Button("Proceed to Review", id="proceed-to-review-button", n_clicks=0, color="success", className="d-block mx-auto"),
    ], className="align-items-center justify-content-center container p-5" )
])

@callback(
    Output("model-selected-text", "children"),
    Input("Logistic-menu-item", "n_clicks"),
    Input("gemini-menu-item", "n_clicks"),

    prevent_initial_call=True,
)
def display_selected_model(scikit_click, gemini_click):
    trigger = ctx.triggered_id

    message = "No model selected."

    if trigger == "Logistic-menu-item":
        message = "Logistic Regression model selected."
    if trigger == "gemini-menu-item":
        message = "Google Gemini model selected."

    return message


@callback(
    [Output("error-message", "children", allow_duplicate=True),
     Output("input-api-key-alert", "is_open"),
     Output("verified-api-key-store", "data"),
     Output("success-toast", "is_open")],
    [Input("input-api-key", "value"),
     Input("submit-api-key-button", "n_clicks")],
    prevent_initial_call=True
)
def verify_api_input(input_value, n_clicks):
    if input_value is None:
        raise PreventUpdate
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    error_messages = []
    if n_clicks > 0:
        if len(input_value) != 50:
            error_messages.append("Notion API keys are 50 characters long.")

        if not input_value.startswith("ntn_"):
            error_messages.append("All Notion API keys start with 'ntn_'.")

    if error_messages:
        error_messages = [html.P(message) for message in error_messages]
        return  error_messages, True, None, False,

    if not error_messages:
        # if no error messages then test with API call for SUCCESS
        notion_manager = NotionManager()
        try:
            success = notion_manager.test_notion_connection(input_value)

            # start encryption
            encryption_manager = EncryptionManager()
            encrypted_api = encryption_manager.encrypt_data(input_value)
            # encrypted key has to be stored as a string, not bytes
            encrypted_api_string = encrypted_api.decode()

            if not success:
                error_messages.append("There is an issue connecting to the Notion API. Please check your key.")
                return error_messages, True, None, False,
        except Exception as e:
            error_messages.append(str(e))
            error_messages.append("There is an issue connecting to the Notion API. Please check your key.")
            return error_messages, True, None, False,


    return "", False, encrypted_api_string, True,



@callback(
    [
        Output("url-location", "pathname", allow_duplicate=True),
    ],
    [
        Input("redirect-violations-timer", "n_intervals"),
    ],
    prevent_initial_call=True,
)
def redirect_to_review_violations(n_intervals):
    """Redirect to review violations page."""
    if n_intervals is None:
        raise PreventUpdate
    print(">>> Redirecting user to review violations page...")
    return ["/review-violations"]

@callback(
    [
        Output("error-message", "children",allow_duplicate=True),
        Output("input-api-key-alert", "is_open", allow_duplicate=True),
        Output("url-location", "pathname", allow_duplicate=True),
        Output("redirect-violations-timer", "disabled"),
        Output("model-selected-store", "data"),
    ],
    [
        Input("Logistic-menu-item", "n_clicks"),
        Input("gemini-menu-item", "n_clicks"),
        Input("proceed-to-review-button", "n_clicks"),
    ],
    [
        State("model-selected-store", "data"),
        State("verified-api-key-store", "data"),
    ],
    prevent_initial_call=True,

)
def redirect_to_selected_model_review(n_clicks_scikit, n_clicks_gemini, n_clicks_proceed, model_selected, api_key):
    """ Depending on the model selected, redirect to the appropriate review violations page."""

    trigger = ctx.triggered_id

    error_message = ""
    show_alert = False
    redirect_path = no_update
    timer_disabled = True
    model_choice = model_selected

    if not api_key and trigger == "redirect-to-review-button":
        error_message = f"Please verify your Notion API key before proceeding."
        show_alert = True
        return error_message, show_alert, redirect_path, timer_disabled, model_choice

    if trigger == "Logistic-menu-item" and (n_clicks_scikit is not None and n_clicks_scikit > 0):
        model_choice = "scikit-learn"
        print(f">>> User selected Sci-kit Learn model")
        return error_message, show_alert, redirect_path, timer_disabled, model_choice

    elif trigger == "gemini-menu-item" and (n_clicks_gemini is not None and n_clicks_gemini > 0):
        model_choice = "gemini"
        print(f">>> User selected Google Gemini model")
        return error_message, show_alert, redirect_path, timer_disabled, model_choice

    elif trigger == "proceed-to-review-button" and (n_clicks_proceed is not None and n_clicks_proceed > 0):
        # If no model is selected, prevent update
        if not model_choice:
            error_message = "No model selected. Please select a model."
            show_alert = True
            return error_message, show_alert, redirect_path, timer_disabled, model_choice

        if model_choice == "gemini":
            redirect_path = "/ai-violations-review"
        else:
            redirect_path = "/review-violations"
        print(f">>> Redirecting user to {redirect_path} ...")
        timer_disabled = False
        return error_message, show_alert, redirect_path, timer_disabled, model_choice

    return error_message, show_alert, redirect_path, timer_disabled, model_choice
