from compliance_engine import ComplianceEngine
import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

def create_app():
    """Create and configure the Dash application"""
    app = Dash(
        __name__,
        use_pages=True,
        external_stylesheets=[
            dbc.themes.LUMEN,
            "https://use.fontawesome.com/releases/v5.15.4/css/all.css"
        ]
    )
    app.config.suppress_callback_exceptions = True
    return app

def run_model_setup():
    """Set up requirements for models"""
    print(">>> Running Compliance Engine...")
    compliance_engine = ComplianceEngine()
    # Train model
    soc2_classifier, soc2_vectorizer = compliance_engine.SOC2_classifier()
    compliance_engine.train_SOC2_classifier(soc2_classifier, soc2_vectorizer)
    # Load SOC2 Classifier
    soc2_classifier_loaded, soc2_vectorizer_loaded = compliance_engine.load_trained_model("SOC2_Model")

def create_navbar():
    """ Create Dash navigation bar"""
    navbar = dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col(html.I(className="fas fa-shield-alt me-2", style={'font-size': '20px'})),
                dbc.Col(dbc.NavbarBrand("NotionArmor", className="ms-2 fw-bold")),
            ],
            align="center",
            className="g-0",
            ),

            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Home", href="/")),
                    dbc.NavItem(dbc.NavLink("Violations Review", href="/review-violations")),
                    dbc.NavItem(dbc.NavLink("AI Violations Review", href="/ai-violations-review")),
                    dbc.NavItem(dbc.NavLink("Dashboard", href="/dashboard")),
                ],
                className="ms-auto",
                navbar=True
                ),
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ]),
        color="primary",
        dark=True,
        sticky="top",
        className="mb-5",
    )
    return navbar

def main():
    run_model_setup()
    print(f"Dash Refreshing \n Dash Version: {dash.__version__}")

    app = create_app()

    app.layout = html.Div([
        create_navbar(),
        dbc.Container([
            dash.page_container
        ])
    ])


    try:
        app.run_server(debug=False)
    except Exception as e:
        print(f"Failed to start Dash server: {e}")

if __name__ == "__main__":
    main()