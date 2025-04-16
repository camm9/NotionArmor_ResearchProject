import dash
from ai_agent import AI_Agent
from compliance_engine import ComplianceEngine
from database_manager import DatabaseManager
from dash.exceptions import PreventUpdate
from dash import html, dcc, callback, Input, Output, ctx, ALL, no_update, State
import dash_bootstrap_components as dbc
from notion_client_manager import NotionManager
from encryption_manager import EncryptionManager

""" Dashboard page in Dash for Front End
 Violation Stats:
 - Total number of violations (and percentages)
 - Most common violation found
 - Pages with highest violation counts
 Model Stats:
 - User feedback correction rate (how often is the user correcting the model)
 System Health:
 
 Requirements:
 - SQLite DB for stat storage -> database_manager.py
 """

dash.register_page(__name__)

layout = html.Div([
    dcc.Store(id="verified-api-key-store", storage_type="session"),
    dbc.Container([
        html.H1('Dashboard'),

        dcc.Loading(id="loading-dashboard",
                    children=[
                        dbc.Row([
                            dbc.Card([
                                dbc.CardBody([
                                ], id='review-data-card',
                                    className='card-body', )],
                            )
                        ]),
                    ],
                    type="cube",
                    className="position-fixed top-50 start-50 translate-middle"
                    ),

        dbc.Row([
            dbc.Col([
                dbc.Button("Refresh Dashboard", id="refresh-dashboard", color="primary"),
            ])
        ])
    ]),

])


# refresh data
@callback(
    [Output('review-data-card', 'children')],
    [Input('refresh-dashboard', 'n_clicks'),
     Input("verified-api-key-store", "data")],
)
def refresh_dashboard(n_clicks, encrypted_api_key):
    """Refresh Dashboard Data"""

    # init
    compliance_engine= ComplianceEngine()
    ai_agent = AI_Agent()
    encryption_manager = EncryptionManager()

    # turn the stored string back into bytes for decryption
    api_key = encryption_manager.decrypt_data(encrypted_api_key).decode()
    notion_manager = NotionManager(api_key=api_key)

    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    try:
        db_manager = DatabaseManager()
        summary = db_manager.get_review_summary()
        ai_analysis_stats = db_manager.get_ai_analysis_stats(notion_manager)
        ai_verification_stats = db_manager.get_ai_verification_stats()
        scikit_verification_stats = db_manager.get_scikit_verification_stats()
        logistic_regression_stats = db_manager.get_logistic_regression_stats(notion_manager)
        dict_of_users, dict_of_bots = notion_manager.get_list_of_users_and_bots()
        number_of_users = len(dict_of_users)
        number_of_bots = len(dict_of_bots)


        return [[
            # General Stats Card
            dbc.Card([
                dbc.CardHeader(html.H4("Review Summary"), className="card-title", style={"background-color": "lightblue"}),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Total Reviews Completed", className="card-subtitle"),
                            html.H3(summary["review_count"], className="card-text"),
                        ]),
                        dbc.Col([
                            html.H5("Total Violations Found", className="card-subtitle"),
                            html.H3(summary["violation_count"], className="card-text"),
                        ])
                    ])
                ])
            ], className="mt-3 mb-6",  style={"border": "1px solid #ffebee", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "border-radius": "8px"}),
            # AI Analysis Card
            dbc.Card([
                dbc.CardHeader(html.H4("AI Analysis Review Summary"), className="card-title", style={"background-color": "#a66e9c"}),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Total Violations Found", className="card-subtitle"),
                            html.H3(ai_analysis_stats["violation_count"], className="card-text"),
                        ]),
                        dbc.Col([
                            html.H5("Violations Found By Category", className="card-subtitle"),
                            html.H3(format_dict_for_display(ai_analysis_stats["category_counts"]),
                                    className="card-text"),
                        ]),
                        dbc.Col([
                            html.H5("Violations Found By Severity", className="card-subtitle"),
                            html.H3(format_dict_for_display(ai_analysis_stats["severity_counts"]),
                                    className="card-text"),
                        ]),
                        dbc.Col([
                            html.H5("Pages with the Most Violations Found", className="card-subtitle"),
                            html.H3(format_dict_for_display(ai_analysis_stats["page_counts"], limit=5),
                                    className="card-text"),
                        ]),
                    ])
                ])
            ], className="mt-3 mb-6", style={"border": "1px solid #ffebee", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "border-radius": "8px"}),
            # Logistic Regression Initial Review Stats
            dbc.Card([
                dbc.CardHeader(html.H4("Logistic Regression Analysis Summary"), className="card-title",
                               style={"background-color": "#ffffa4"}),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Total Violations Found", className="card-subtitle"),
                            html.H3(logistic_regression_stats.get("violation_count", 0), className="card-text"),
                        ]),
                        dbc.Col([
                            html.H5("Violations Found By Category", className="card-subtitle"),
                            html.H3(format_dict_for_display(logistic_regression_stats.get("category_counts", {})),
                                    className="card-text"),
                        ]),
                        dbc.Col([
                            html.H5("Pages with the Most Violations Found", className="card-subtitle"),
                            html.H3(format_dict_for_display(logistic_regression_stats.get("page_counts", {}), limit=5),
                                    className="card-text"),
                        ]),
                    ])
                ])
            ], className="mt-3 mb-6",
                style={"border": "1px solid #ffebee", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                       "border-radius": "8px"}),
            # Remediation Card
            dbc.Card([
                dbc.CardHeader(html.H4("Remediation Stats"), className="card-title", style={"background-color": "#ffa4ef"}),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Successful User Applied Redactions", className="card-subtitle"),
                            html.H3(summary["successful_user_redactions"], className="card-text"),
                        ]),
                        dbc.Col([
                            html.H5("Unsuccessful User Applied Redactions", className="card-subtitle"),
                            html.H3(summary["unsuccessful_user_redactions"], className="card-text"),
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Successfully AI Applied Remediations", className="card-subtitle"),
                            html.H3(summary.get("successful_ai_remediations", 0), className="card-text"),
                        ]),
                        dbc.Col([
                            html.H5("Unsuccessful AI Applied Remediations", className="card-subtitle"),
                            html.H3(summary.get("unsuccessful_ai_remediations",0), className="card-text"),
                        ]),
                    ]),
                ])
            ], className="mt-3 mb-6", style={"border": "1px solid #ffebee", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "border-radius": "8px"}),
            # AI User Verification Stats
            dbc.Card([
                dbc.CardHeader(html.H4(f"Gemini AI User Verification Stats"), className="card-title", style={"background-color": "#a4ffcb"}),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Total Verifications by User", className="card-subtitle"),
                            html.H3(ai_verification_stats["total_verifications"], className="card-text"),
                        ]),
                        dbc.Col([
                            html.H5("Accepted Without Changes (True Positives)", className="card-subtitle"),
                            html.Div([
                                html.H3(f"{ai_verification_stats['correct_predictions']['count']} ", className="card-text me-2"),
                                html.Span(f"({ai_verification_stats['correct_predictions']['percentage']}%)",
                                          className="text-muted")
                            ], className="d-flex align-items-baseline")
                        ]),
                        dbc.Col([
                            html.H5("Marked as Compliant (False Positives)", className="card-subtitle"),
                            html.Div([
                                html.H3(f"{ai_verification_stats['false_positives']['count']} ", className="card-text me-2"),
                                html.Span(f"({ai_verification_stats['false_positives']['percentage']}%)",
                                          className="text-muted")
                            ], className="d-flex align-items-baseline")
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Categories Modified (Partial True Positives)", className="card-subtitle"),
                            html.Div([
                                html.H3(f"{ai_verification_stats['partially_correct_predictions']['count']} ", className="card-text me-2"),
                                html.Span(f"({ai_verification_stats['partially_correct_predictions']['percentage']}%)",
                                          className="text-muted")
                            ], className="d-flex align-items-baseline")
                        ]),
                        dbc.Col([
                            html.H5("Severity Modified (Incorrect Severities Assigned)", className="card-subtitle"),
                            html.Div([
                                html.H3(f"{ai_verification_stats['severity_modified']['count']}", className="card-text me-2"),
                                html.Span(f"({ai_verification_stats['severity_modified']['percentage']}%)",
                                          className="text-muted")
                            ], className="d-flex align-items-baseline")
                        ]),
                    ]),
                ])
            ], className="mt-3 mb-6", style={"border": "1px solid #ffebee", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "border-radius": "8px"}),
            # User Verification of Scikit Learn models card
            dbc.Card([
                dbc.CardHeader(html.H4(f"Logistic Regression Model Verification Stats"), className="card-title", style={"background-color": "#a4dcff"}),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Total Verifications by User", className="card-subtitle"),
                            html.H3(scikit_verification_stats["total_verifications"], className="card-text"),
                        ]),
                        dbc.Col([
                            html.H5("Accepted Without Changes (True Positives/Negatives)", className="card-subtitle"),
                            html.Div([
                                html.H3(f"{scikit_verification_stats['correct_predictions']['count']} ",
                                        className="card-text me-2"),
                                html.Span(f"({scikit_verification_stats['correct_predictions']['percentage']}%)",
                                          className="text-muted")
                            ], className="d-flex align-items-baseline")
                        ]),
                        dbc.Col([
                            html.H5("Marked as Compliant (False Positives)", className="card-subtitle"),
                            html.Div([
                                html.H3(f"{scikit_verification_stats['false_positives']['count']} ",
                                        className="card-text me-2"),
                                html.Span(f"({scikit_verification_stats['false_positives']['percentage']}%)",
                                          className="text-muted")
                            ], className="d-flex align-items-baseline")
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Categories Modified (Partial True Positives)", className="card-subtitle"),
                            html.Div([
                                html.H3(f"{scikit_verification_stats['partially_correct_predictions']['count']} ",
                                        className="card-text me-2"),
                                html.Span(
                                    f"({scikit_verification_stats['partially_correct_predictions']['percentage']}%)",
                                    className="text-muted")
                            ], className="d-flex align-items-baseline")
                        ]),
                    ]),
                ])
            ], className="mt-3 mb-6", style={"border": "1px solid #ffebee", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "border-radius": "8px"}),
            # User Audit Card
            dbc.Card([
                dbc.CardHeader(html.H4("User Privileges"), className="card-title", style={"background-color": "#ffd8a4"}),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Number of Users", className="card-subtitle"),
                            html.H3(number_of_users)
                        ]),
                        dbc.Col([
                            html.H5("User List", className="card-subtitle"),
                            html.P(html.Ul([html.Li(user['email']) for user in dict_of_users.values()], className="list-unstyled"))
                        ])
                    ])
                ])
            ], className="mt-3 mb-6", style={"border": "1px solid #ffebee", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "border-radius": "8px"}),
            # External Integrations
            dbc.Card([
                dbc.CardHeader(html.H4("External Integrations"), className="card-title", style={"background-color": "#d8ffa4"}),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Number of Service Accounts", className="card-subtitle"),
                            html.H3(number_of_bots),
                        ]),
                        dbc.Col([
                            html.H5("Application Integration List", className="card-subtitle"),
                            html.P(html.Ul([html.Li(bot['bot_name']) for bot in dict_of_bots.values()],className="list-unstyled")),
                        ])
                    ])
                ])
            ], className="mt-3 mb-6", style={"border": "1px solid #ffebee", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "border-radius": "8px"}),
        ]]



    except Exception as e:
        print(f"Error refreshing Dashboard: {e}")

        return [[
            # General Stats Card
            dbc.Card([
                dbc.CardHeader(html.H4("Error Summary"), className="card-title"),
                dbc.CardBody([
                    dbc.Col([
                        html.H3("Error loading data", className="card-text"),
                    ])

                ])
            ]),

        ]]


def format_dict_for_display(data_dict, limit=None):
    """Format the dictionary  response into a readable string for display"""
    if not data_dict:
        return "None"

    items = list(data_dict.items())
    total_items = len(data_dict)
    showing_limited = False

    if limit is not None and total_items > limit:
        displayed_items = items[:limit]
        showing_limited = True
    else:
        displayed_items = items

    formatted_items = []
    for key, value in displayed_items:
        # Check if the key contains a URL (from our page_counts dictionary)
        if "(" in key and ")" in key and "http" in key:
            # Extract the URL from between parentheses
            url_start = key.find("(") + 1
            url_end = key.find(")")
            url = key[url_start:url_end]

            # Create the display text (without the URL part)
            display_text = f"{key[:key.find('(')].strip()}: {value}"

            # Create a clickable link
            formatted_items.append(html.P([
                display_text, " ",
                html.A("Open", href=url, target="_blank")
            ]))
        else:
            # Regular item with no URL
            formatted_items.append(html.P(f"{key}: {value}"))

    # if there are more than 5 items, place a "show more" button
    if limit is not None and total_items > limit:
        return html.Div([
            html.Ul(formatted_items, className="list-unstyled"),
            html.Div(f"Showing {limit} of {total_items} items",
                     className="text-muted small")
        ])
    else:
        return html.Ul(formatted_items, className="list-unstyled") if items else "None"
