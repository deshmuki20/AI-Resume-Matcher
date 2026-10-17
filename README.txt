AI RESUME AND JOB DESCRIPTION MATCHER - FLASK + FREE GEMINI VERSION
===================================================================
Uses the FREE Google Gemini API (no credit card or payment needed).
DECISION SUPPORT ONLY: it never decides to hire or reject anyone.

FILES
-----
resume_matcher.py   - the whole application (only Python file)
requirements.txt    - packages to install
README.txt          - this guide
.env                - your secret API key (never share it)

STEP 1 - GET A FREE GEMINI API KEY
----------------------------------
1. Open https://aistudio.google.com/apikey in your browser.
2. Sign in with your Google account.
3. Click "Create API key" and copy the key.
(Free tier has limits per minute/day. If you hit a limit, wait and try again.)

STEP 2 - SET UP PYCHARM
-----------------------
1. PyCharm -> New Project -> Virtualenv -> Create.
2. Add the 4 files (right-click project -> New -> File; paste contents).
3. Open Terminal (Alt+F12). If the line starts with "PS", type: cmd
   Make sure "(venv)" shows. If not:
       python -m venv venv
       venv\Scripts\activate.bat
4. Install packages:
       pip install -r requirements.txt

STEP 3 - ADD YOUR KEY
---------------------
Open .env and set:
       GOOGLE_API_KEY=your-copied-key
No quotes, no spaces. Save with Ctrl+S.

STEP 4 - RUN
------------
In the Terminal:   python resume_matcher.py
(or open resume_matcher.py and click the green Run button)
Open your browser at:  http://127.0.0.1:5000

STEP 5 - USE
------------
1. Upload a resume (PDF or TXT).
2. Paste the job description.
3. Click "Analyze Resume" and wait 15-40 seconds.
4. Click "Download Report (TXT)" to save the report.

STOP: press Ctrl+C in the Terminal (or the red Stop button in PyCharm).
After changing .env, stop and run the app again.

PRIVACY NOTE
------------
On the free Gemini tier, Google may use requests to improve its products.
Use sample or anonymised resumes, not real candidates' personal data.

TROUBLESHOOTING
---------------
"No API key found"       -> check .env, save it, restart the app.
"API key rejected"       -> copy a fresh key from Google AI Studio.
"free-tier limit"        -> wait one minute and try again; limits reset daily.
"model was not found"    -> in .env set GEMINI_MODEL to another model name shown
                            in Google AI Studio, save, restart.
"No module named ..."    -> the venv is not active, or pip install was skipped.