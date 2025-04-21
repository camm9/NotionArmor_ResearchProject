

### NotionArmor
Description: This application will go through the linked pages in your Notion workspace and return strings of data. Users can choose between using 
a machine learning model or Google Gemini to detect SOC2 violations. 
Users can review the findings of the model and verify their accuracy. The user feedback can then
be sent back to the model for re-training (Logistic Regression model only). Users can also choose to remediate the violations with text redactions.
Statistics regarding the models' successes can be viewed on the dashboard as well as what users and service accounts have access to the Notion
workspace.
<br> <br>
By: <br>
Catherine Methven 300361000 - methvenc@student.douglascollege.ca <br>

This student project was made CSIS4495 - Applied Research Project at Douglas College in New Westminster, Canada.

# Instructions to Set Up Notion Workspace
Step 1. Log into your Notion account and go to the Developer Portal. Select the Integrations tab. <br>
Step 2. Create a new integration <br>
![Create a new integration on the Integrations page of settings ](/Misc/Screenshots/notion_integrations.jpg)
Step 3. Name the integration and set as an 'internal' type
![Give a name and Internal type to the integration](/Misc/Screenshots/save_internal_integration.jpg)
Step 4. Copy your API key, you'll need this later. Select "Read content", "Update content", and "Insert content" from Capabilities
![Here is your API key, copy this, and select only "Read content"](/Misc/Screenshots/api_settings.jpg)
Step 5. Navigate to the page you wish to examine, click the ellipsis menu, select 'Connections' and the integration you created.
![Enable integration on pages you wish to scan](/Misc/Screenshots/page_integration.jpg)
<br>
#### Note: Application only works with pages and not Wikis. Please connect to a page. 
You are now ready to run the application.

# Instructions to Run
#### Note: For your API key to function you must have enabled connections with your Notion workspaces.

Step 1. Clone repo 

<br>
Step 2. Create a .env file inside cloned folder <br>

> echo "GEMINI_API=ADD_VALUE_HERE" >> .env

Gemini API keys can be obtained for free at https://ai.google.dev/gemini-api/docs/api-key <br>

Step 3. Navigate to Implementation folder


>cd /Implementation <br>


Step 4. Create a virtual environment using Python 3.11<br>


> Mac instructions<br>
>/usr/local/bin/python3.11 -m venv venv <br>
source venv/bin/activate <br>
> Windows instructions <br>
> python -m venv venv <br>
> ./venv/Scripts/activate <br>

 

Step 5. Install Required packages <br>


> pip install -r requirements.txt <br>
python /backend/app.py 

<br>

Step 6. Visit in your browser http://127.0.0.1:8050

# Features
- Select a model for analysis & predict likelihood of SOC2 violation on a list of strings found in Notion workspace
- Give user feedback on the found violations
- Remediate violations
- Re-train model with user data (Logistic Regression model only)
- Review stats from predictions and remediations, and user and service account access
