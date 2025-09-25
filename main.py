import time
import requests, re, io, json, smtplib, ast 
from PyPDF2 import PdfReader
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import OpenAI
from gspread_formatting import format_cell_range, CellFormat, TextFormat, Color
from googleapiclient.discovery import build
from fastapi import FastAPI, Request, BackgroundTasks

# --- FASTAPI APP ---
app = FastAPI()

# --- DEEPSEEK CLIENT ---
client = OpenAI(
    api_key="sk-32faa3252eb14792b99fdb438c724f8f",
    base_url="https://api.deepseek.com"
)

# --- PARSE FORM INPUTS ---
def parse_form_input(data, creds_path="analog-opus-435209-q6-a06efa6f2492.json"):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    return {
        "folder": data.get("Link to Google Drive Resumes Folder (Make sure it's public)", ""),
        "email": data.get("Email address", ""),
        "job": data.get("Job Description", ""),
        "filters": data.get("Filtering Conditions", ""),
        "creds": creds
    }

# --- FILE UTILS ---
def extract_folder_id(url): return re.search(r"/folders/([a-zA-Z0-9_-]+)", url).group(1)
def extract_file_id(url): return re.search(r"/d/([a-zA-Z0-9_-]+)", url).group(1)
def file_available(url): return requests.get(url).status_code != 404
def download_pdf(file_id): return io.BytesIO(requests.get(f"https://drive.google.com/uc?export=download&id={file_id}").content)

def extract_text(pdf_bytes):
    try:
        reader = PdfReader(pdf_bytes)
        text = ""
        for i, page in enumerate(reader.pages):
            t = page.extract_text()
            if t:
                text += f"\n--- Page {i+1} ---\n{t.strip()}\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"❌ Failed to read PDF: {e}")

def get_file_links(folder_url):
    folder_id = extract_folder_id(folder_url)
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        "analog-opus-435209-q6-a06efa6f2492.json",
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=creds)
    files, page_token = [], None
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    while True:
        response = service.files().list(
            q=query,
            spaces='drive',
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)
        if not page_token: break
    return [f"https://drive.google.com/file/d/{f['id']}/view?usp=sharing" for f in files]

# --- DEEPSEEK INTERACTION ---
def build_prompt(cv, job, filters):
    return [
        {"role": "system", "content":
"""You are a highly experienced and strict technical recruiter. Your task is to evaluate candidates *objectively and consistently* using only the provided resume, job description, and mandatory conditions. Always respond with valid JSON only — no explanation, no markdown, and no additional formatting.

Be extremely selective. Assume hundreds of candidates apply and only the top few should pass. Your evaluation must be reproducible and based on measurable criteria. Avoid vague or subjective language, and apply the rules exactly as described.

Scoring must be precise:
• 0–19.99 → 'Does Not Meet Criteria'
• 20–39.99 → 'Not Ideal Fit'
• 40–59.99 → 'Consider for Review'
• 60–79.99 → 'High Potential'
• 80–100 → 'Top Candidate'

Skills evaluation must only consider exact or equivalent matches to the skills listed in the job description.

If any of the mandatory conditions are not satisfied, deduct significantly from the score. The output must follow the structure exactly."""},

{"role": "user", "content": f"""
Evaluate the following candidate strictly.

--- Resume ---
{cv}

--- Job Description ---
{job}

--- Mandatory Conditions ---
{filters}

🎯 Output EXACTLY this JSON object:

{{
  "candidate_name": "<string>",
  "job_title": "<job title from the resume>",
  "education": "<degree, major, university>",
  "graduation_year": <year>,
  "years_of_experience": "<real work experience, internships separate>",

  "skills_match": <0–100>,
  "education_relevance": <0–100>,
  "experience_relevance": <0–100>,
  "score": <0–100>,
  "decision": "<based on ranges>",

  "conditions": "<evaluate mandatory conditions>",
  "strengths": "<brief>",

  "matched_skills": [list],
  "missing_skills": [list],
  "extra_related_skills": [list],
  "extra_unrelated_skills": [list],

  "phone": "<from resume or null>",
  "email_address": "<from resume or null>"
}}
"""}
    ]

def evaluate(cv_text, job, filters):
    try:
        prompt = build_prompt(cv_text, job, filters)
        res = client.chat.completions.create(model="deepseek-chat", messages=prompt, temperature=0)
        return res.choices[0].message.content
    except Exception as e:
        return f"❌ API error: {e}"

# --- JSON PARSING ---
def parse_response(raw_response):
    try:
        json_str = re.search(r'\{.*\}', raw_response, re.DOTALL).group(0)
        try: data = json.loads(json_str)
        except json.JSONDecodeError: data = ast.literal_eval(json_str)
        return [
            data.get("candidate_name", ""), data.get("job_title", ""), data.get("education", ""), data.get("graduation_year", ""),
            data.get("years_of_experience", ""), data.get("skills_match", ""), data.get("education_relevance", ""),
            data.get("experience_relevance", ""), data.get("score", ""), data.get("decision", ""), data.get("conditions", ""),
            data.get("strengths", ""),
            ", ".join(data.get("matched_skills", [])), ", ".join(data.get("missing_skills", [])),
            ", ".join(data.get("extra_related_skills", [])), ", ".join(data.get("extra_unrelated_skills", [])),
            f"'{data.get('phone', '')}", data.get("email_address", "")
        ]
    except Exception as e:
        print(f"❌ JSON parsing error: {e}")
        return ["❌ JSON error"] + [""] * 25

# --- SHEET FORMATTING ---
def format_and_sort_sheet(ws):
    sheet_id = ws._properties['sheetId']
    ws.spreadsheet.batch_update({
        "requests": [{
            "sortRange": {
                "range": {"sheetId": sheet_id, "startRowIndex": 2},
                "sortSpecs": [{"dimensionIndex": 9, "sortOrder": "DESCENDING"}]
            }
        }]
    })
    format_cell_range(ws, '2:2', CellFormat(textFormat=TextFormat(bold=True)))
    ws.freeze(rows=2)

# --- EMAIL NOTIFICATION ---
def send_email(to_email, sheet_link, expired=[]):
    msg = MIMEMultipart()
    msg['From'] = "mahmadmahmod87@gmail.com"
    msg['To'] = to_email
    msg['Subject'] = "Resume Evaluation Results"
    body = f"Results: {sheet_link}"
    if expired: body += "\n\nSome files were expired:\n" + "\n".join(expired)
    msg.attach(MIMEText(body, 'plain'))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("mahmadmahmod87@gmail.com", "ccdr sgfa oihn iran")
    server.sendmail("mahmadmahmod87@gmail.com", to_email, msg.as_string())
    server.quit()

# --- MAIN WORKFLOW ---
def run(form_data: dict):
    start = time.time()
    i = parse_form_input(form_data)
    links = get_file_links(i["folder"])
    results, expired = [], []

    for link in links:
        file_id = extract_file_id(link)
        url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        print(f"📄 Processing → {url}")
        if not file_available(url):
            expired.append(url)
            continue
        try:
            pdf = download_pdf(file_id)
            text = extract_text(pdf)
            response = evaluate(text, i["job"], i["filters"])
            with open("raw_llm_outputs.txt", "a", encoding="utf-8") as f:
                f.write(f"\n\n--- {url} ---\n{response}\n")
            results.append([url] + parse_response(response))
        except Exception as e:
            print(f"❌ Error for {url}: {e}")
            results.append([url, f"❌ Error: {e}"] + [""] * 25)

    gc = gspread.authorize(i["creds"])
    sh = gc.create("Resume Evaluation Results")
    ws = sh.sheet1

    category_row = ["Summary"] * 6 + ["Scores"] * 5 + ["Review"] * 2 + ["Skills"] * 4 + ["Contact Info"] * 2
    header_row = [
        "CV URL", "Name", "Job Title", "Education", "Grad Year", "Experience",
        "Skills Match", "Edu Relevance %", "Exp Relevance %", "Overall Score", "Tier",
        "Mandatory Conditions (Evaluation)", "Strengths",
        "Matched Skills", "Missing Skills", "Extra Related Skills", "Extra Unrelated Skills",
        "Phone Number", "Email Address"
    ]
    ws.insert_rows([category_row, header_row], 1)
    ws.append_rows(results, value_input_option="USER_ENTERED")

    format_and_sort_sheet(ws)
    sh.share(i["email"], perm_type='user', role='writer')
    send_email(i["email"], sh.url, expired)
    print(f"🎯 All done in {time.time() - start:.2f} seconds")

# --- FASTAPI ENDPOINTS ---
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/")
async def process_form(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    print("📥 Received:", data)
    background_tasks.add_task(run, data)
    return {
        "status": "processing",
        "email": data.get("Email address"),
        "message": "Resume evaluation started. You will receive an email when it's done."
    }
