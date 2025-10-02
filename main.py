import os
import time
import asyncio
import aiohttp
import json
import smtplib
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache

import PyPDF2
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import AsyncOpenAI
from gspread_formatting import format_cell_range, CellFormat, TextFormat, Color
from googleapiclient.discovery import build
from fastapi import FastAPI, Request, BackgroundTasks
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class ProcessingConfig:
    """Configuration for processing parameters"""
    max_concurrent_downloads: int = 10
    max_concurrent_evaluations: int = 5
    batch_size: int = 50
    timeout: int = 30

# Global config instance
config = ProcessingConfig()

# --- FASTAPI APP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize HTTP session
    app.state.http_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=config.timeout),
        connector=aiohttp.TCPConnector(limit=100)
    )
    yield
    # Shutdown: Close HTTP session
    await app.state.http_session.close()

app = FastAPI(lifespan=lifespan)

# --- DEEPSEEK CLIENT ---
async_client = AsyncOpenAI(
    api_key=os.environ.get("DEEP_SEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# --- CACHED CREDENTIALS ---
@lru_cache(maxsize=1)
def get_credentials():
    """Cache credentials to avoid repeated parsing"""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    service_account_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT"])
    return ServiceAccountCredentials.from_json_keyfile_dict(service_account_info, scope)

# --- OPTIMIZED PARSING ---
def parse_form_input(data: Dict) -> Dict:
    """Optimized form parsing with cached credentials"""
    return {
        "folder": data.get("Link to Google Drive Resumes Folder (Make sure it's public)", ""),
        "email": data.get("Email address", ""),
        "job": data.get("Job Description", ""),
        "filters": data.get("Filtering Conditions", ""),
        "creds": get_credentials()
    }

# --- OPTIMIZED FILE UTILS ---
import re
from io import BytesIO

# Compile regex patterns once
FOLDER_ID_PATTERN = re.compile(r"/folders/([a-zA-Z0-9_-]+)")
FILE_ID_PATTERN = re.compile(r"/d/([a-zA-Z0-9_-]+)")
JSON_PATTERN = re.compile(r'\{.*\}', re.DOTALL)

def extract_folder_id(url: str) -> str:
    match = FOLDER_ID_PATTERN.search(url)
    if not match:
        raise ValueError(f"Invalid folder URL: {url}")
    return match.group(1)

def extract_file_id(url: str) -> str:
    match = FILE_ID_PATTERN.search(url)
    if not match:
        raise ValueError(f"Invalid file URL: {url}")
    return match.group(1)

async def check_file_availability(session: aiohttp.ClientSession, url: str) -> bool:
    """Async file availability check"""
    try:
        async with session.head(url) as response:
            return response.status != 404
    except Exception:
        return False

async def download_pdf_async(session: aiohttp.ClientSession, file_id: str) -> BytesIO:
    """Async PDF download with better error handling"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        async with session.get(url) as response:
            if response.status != 200:
                raise ValueError(f"Failed to download: HTTP {response.status}")
            content = await response.read()
            if len(content) < 100:  # Likely an error page
                raise ValueError("Downloaded content too small")
            return BytesIO(content)
    except Exception as e:
        raise ValueError(f"Download failed: {e}")

def extract_text_optimized(pdf_bytes: BytesIO) -> str:
    """Optimized text extraction with memory management"""
    try:
        reader = PyPDF2.PdfReader(pdf_bytes)
        text_parts = []
        
        # Limit pages to prevent memory issues with large files
        max_pages = min(len(reader.pages), 20)  
        
        for i in range(max_pages):
            try:
                page_text = reader.pages[i].extract_text()
                if page_text and page_text.strip():
                    text_parts.append(f"--- Page {i+1} ---\n{page_text.strip()}")
            except Exception as e:
                logger.warning(f"Failed to extract page {i+1}: {e}")
                continue
        
        # Clear the BytesIO buffer to free memory
        pdf_bytes.close()
        
        return "\n".join(text_parts) if text_parts else "No text extracted"
        
    except Exception as e:
        raise ValueError(f"PDF processing failed: {e}")

async def get_file_links_async(folder_url: str) -> List[str]:
    """Async file listing with pagination optimization"""
    folder_id = extract_folder_id(folder_url)
    creds = get_credentials()
    
    # Use async executor for Google API calls
    def _get_files():
        service = build("drive", "v3", credentials=creds)
        files = []
        page_token = None
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        
        while True:
            try:
                response = service.files().list(
                    q=query,
                    spaces='drive',
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                    pageSize=1000  # Larger page size for fewer API calls
                ).execute()
                
                files.extend(response.get("files", []))
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
            except Exception as e:
                logger.error(f"API error: {e}")
                break
        
        return files[:20]
    
    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    files = await loop.run_in_executor(None, _get_files)
    
    return [f"https://drive.google.com/file/d/{f['id']}/view?usp=sharing" for f in files]

# --- OPTIMIZED DEEPSEEK INTERACTION ---
@lru_cache(maxsize=100)
def build_prompt_cached(job_hash: str, filters_hash: str) -> List[Dict]:
    """Cache prompts for repeated job descriptions"""
    return [
        {"role": "system", "content": """you're an expert HR recruiter with experience in 'job_title' - variable from job description . Your task is to evaluate candidates *objectively and consistently* using only the provided resume, job description, and mandatory conditions. Always respond with valid JSON only — no explanation, no markdown, and no additional formatting.

Be extremely selective. Assume hundreds of candidates apply and only the top few should pass. Your evaluation must be reproducible and based on measurable criteria. Avoid vague or subjective language, and apply the rules exactly as described.

Scoring must be precise:
• 0–19.99 → 'Does Not Meet Criteria'
• 20–39.99 → 'Not Ideal Fit'
• 40–59.99 → 'Consider for Review'
• 60–79.99 → 'High Potential'
• 80–100 → 'Top Candidate'

Skills evaluation must only consider exact or equivalent matches to the skills listed in the job description.

If any of the mandatory conditions are not satisfied, deduct significantly from the score. The output must follow the structure exactly."""}
    ]

async def evaluate_async(cv_text: str, job: str, filters: str) -> str:
    """Async evaluation with caching"""
    try:
        # Create hash for caching
        job_hash = str(hash(job))
        filters_hash = str(hash(filters))
        
        system_prompt = build_prompt_cached(job_hash, filters_hash)
        
        messages = system_prompt + [{
            "role": "user", 
            "content": f"""
Evaluate the following candidate strictly.

--- Resume ---
{cv_text[:10000]}  # Limit text length to prevent API limits

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
"""
        }]
        
        response = await async_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0,
            max_tokens=2000
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"API evaluation error: {e}")
        return f"❌ API error: {e}"

# --- OPTIMIZED JSON PARSING ---
def parse_response_optimized(raw_response: str) -> List[str]:
    """Optimized JSON parsing with better error handling"""
    try:
        # Try to find JSON in response
        json_match = JSON_PATTERN.search(raw_response)
        if not json_match:
            return ["❌ No JSON found"] + [""] * 17
            
        json_str = json_match.group(0)
        
        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback to ast.literal_eval
            import ast
            data = ast.literal_eval(json_str)
        
        # Extract data with defaults
        result = [
            data.get("candidate_name", ""),
            data.get("job_title", ""),
            data.get("education", ""),
            str(data.get("graduation_year", "")),
            data.get("years_of_experience", ""),
            str(data.get("skills_match", "")),
            str(data.get("education_relevance", "")),
            str(data.get("experience_relevance", "")),
            str(data.get("score", "")),
            data.get("decision", ""),
            data.get("conditions", ""),
            data.get("strengths", ""),
            ", ".join(data.get("matched_skills", [])),
            ", ".join(data.get("missing_skills", [])),
            ", ".join(data.get("extra_related_skills", [])),
            ", ".join(data.get("extra_unrelated_skills", [])),
            f"'{data.get('phone', '')}" if data.get('phone') else "",
            data.get("email_address", "")
        ]
        
        return result
        
    except Exception as e:
        logger.error(f"JSON parsing error: {e}")
        return ["❌ JSON error"] + [""] * 17

# --- OPTIMIZED SHEET OPERATIONS ---
def create_and_format_sheet(creds, results: List[List[str]]) -> gspread.Spreadsheet:
    """Optimized sheet creation with batch operations"""
    gc = gspread.authorize(creds)
    sh = gc.create("Resume Evaluation Results")
    ws = sh.sheet1
    
    # Prepare all data at once
    category_row = ["Summary"] * 6 + ["Scores"] * 5 + ["Review"] * 2 + ["Skills"] * 4 + ["Contact Info"] * 2
    header_row = [
        "CV URL", "Name", "Job Title", "Education", "Grad Year", "Experience",
        "Skills Match", "Edu Relevance %", "Exp Relevance %", "Overall Score", "Tier",
        "Mandatory Conditions (Evaluation)", "Strengths",
        "Matched Skills", "Missing Skills", "Extra Related Skills", "Extra Unrelated Skills",
        "Phone Number", "Email Address"
    ]
    
    # Insert all data in one batch
    all_data = [category_row, header_row] + results
    ws.update(f'A1:S{len(all_data)}', all_data, value_input_option="USER_ENTERED")
    
    # Format sheet
    sheet_id = ws._properties['sheetId']
    
    # Batch all formatting operations
    batch_requests = []
    
    # Merge cells for categories
    merge_ranges = [(0, 6), (6, 11), (11, 13), (13, 17), (17, 19)]
    for start, end in merge_ranges:
        batch_requests.append({
            "mergeCells": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                    "startColumnIndex": start,
                    "endColumnIndex": end
                },
                "mergeType": "MERGE_ALL"
            }
        })
    
    # Sort by overall score (column 10, index 9)
    batch_requests.append({
        "sortRange": {
            "range": {"sheetId": sheet_id, "startRowIndex": 2},
            "sortSpecs": [{"dimensionIndex": 9, "sortOrder": "DESCENDING"}]
        }
    })
    
    # ===== ADD CONDITIONAL FORMATTING =====
    # Define tier colors and score ranges
    tiers = {
        "Top Candidate": {"color": {"red": 0.6, "green": 1, "blue": 0.6}, "min": 80, "max": 100},
        "High Potential": {"color": {"red": 0.8, "green": 1, "blue": 0.8}, "min": 60, "max": 79.99},
        "Consider For Review": {"color": {"red": 1, "green": 1, "blue": 0.6}, "min": 40, "max": 59.99},
        "Not Ideal Fit": {"color": {"red": 1, "green": 0.9, "blue": 0.5}, "min": 20, "max": 39.99},
        "Does Not Meet The Criteria": {"color": {"red": 1, "green": 0.8, "blue": 0.8}, "min": 0, "max": 19.99}
    }
    
    # Add conditional formatting for Tier (column K, index 10) and Score (column J, index 9)
    for idx, (tier, info) in enumerate(tiers.items()):
        # Format for Tier column (text match)
        batch_requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{"sheetId": sheet_id, "startRowIndex": 2, "startColumnIndex": 10, "endColumnIndex": 11}],
                    "booleanRule": {
                        "condition": {"type": "TEXT_EQ", "values": [{"userEnteredValue": tier}]},
                        "format": {"backgroundColor": info["color"]}
                    }
                },
                "index": idx
            }
        })
        
        # Format for Score column (number range)
        batch_requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{"sheetId": sheet_id, "startRowIndex": 2, "startColumnIndex": 9, "endColumnIndex": 10}],
                    "booleanRule": {
                        "condition": {
                            "type": "NUMBER_BETWEEN",
                            "values": [
                                {"userEnteredValue": str(info["min"])},
                                {"userEnteredValue": str(info["max"])}
                            ]
                        },
                        "format": {"backgroundColor": info["color"]}
                    }
                },
                "index": idx + 5
            }
        })
    
    # Execute all requests in one batch
    ws.spreadsheet.batch_update({"requests": batch_requests})
    
    # Apply formatting
    category_formats = [
        ("A1:F1", Color(1, 0.95, 0.85)),
        ("G1:K1", Color(0.7, 0.95, 0.7)),
        ("L1:M1", Color(0.85, 0.92, 0.98)),
        ("N1:Q1", Color(0.95, 0.95, 0.75)),
        ("R1:S1", Color(0.88, 0.88, 0.98))
    ]
    
    for cell_range, bg_color in category_formats:
        format_cell_range(ws, cell_range, CellFormat(
            textFormat=TextFormat(bold=True),
            horizontalAlignment='CENTER',
            backgroundColor=bg_color
        ))
    
    format_cell_range(ws, '2:2', CellFormat(textFormat=TextFormat(bold=True)))
    ws.freeze(rows=2)
    
    return sh

# --- OPTIMIZED EMAIL ---
async def send_email_async(to_email: str, sheet_link: str, expired: List[str] = None):
    """Async email sending"""
    def _send_email():
        try:
            msg = MIMEMultipart()
            msg['From'] = os.environ.get("EMAIL_USER")
            msg['To'] = to_email
            msg['Subject'] = "Resume Evaluation Results"
            
            body = f"Results: {sheet_link}"
            if expired:
                body += f"\n\n{len(expired)} files were expired or inaccessible."
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(os.environ.get("EMAIL_USER"), os.environ.get("EMAIL_PASS"))
                server.sendmail(os.environ.get("EMAIL_USER"), to_email, msg.as_string())
            
            logger.info(f"Email sent to {to_email}")
        except Exception as e:
            logger.error(f"Email error: {e}")
    
    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _send_email)


async def send_feedback_email_async(to_email: str):
    """Send feedback form link after 1 hour"""
    await asyncio.sleep(3600) 

    def _send_email():
        try:
            msg = MIMEMultipart()
            msg['From'] = os.environ.get("EMAIL_USER")
            msg['To'] = to_email
            msg['Subject'] = "We’d love your feedback on the Resume Evaluation Service"

            body = (
                "Hello,\n\n"
                "Thank you for using the Resume Evaluation Service! 🙏\n\n"
                "Your experience matters to us, and we’d love to hear your thoughts.\n\n"
                "👉 Please share your feedback here: https://forms.gle/xtx5ZAidEKnVMCAT6\n\n"
                "Best regards,\n"
                "The Resume Evaluation Team"
            )

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(os.environ.get("EMAIL_USER"), os.environ.get("EMAIL_PASS"))
                server.sendmail(os.environ.get("EMAIL_USER"), to_email, msg.as_string())

            logger.info(f"✅ Feedback email sent to {to_email}")
        except Exception as e:
            logger.error(f"Feedback email error: {e}")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _send_email)

# --- MAIN OPTIMIZED WORKFLOW ---
async def run_optimized(form_data: Dict):
    """Main optimized workflow with async processing"""
    start_time = time.time()
    
    try:
        # Parse input
        inputs = parse_form_input(form_data)
        
        # Get file links
        logger.info("Fetching file links...")
        links = await get_file_links_async(inputs["folder"])
        logger.info(f"Found {len(links)} PDF files")
        
        if not links:
            logger.warning("No PDF files found")
            return
        
        # Process files in concurrent batches
        results = []
        expired = []
        
        # Create semaphores to limit concurrency
        download_semaphore = asyncio.Semaphore(config.max_concurrent_downloads)
        eval_semaphore = asyncio.Semaphore(config.max_concurrent_evaluations)
        
        async def process_file(session: aiohttp.ClientSession, link: str) -> Optional[List[str]]:
            """Process a single file"""
            try:
                file_id = extract_file_id(link)
                
                # Check availability
                if not await check_file_availability(session, link):
                    expired.append(link)
                    return None
                
                # Download PDF
                async with download_semaphore:
                    pdf_bytes = await download_pdf_async(session, file_id)
                
                # Extract text in thread pool
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, extract_text_optimized, pdf_bytes)
                
                # Evaluate with AI
                async with eval_semaphore:
                    response = await evaluate_async(text, inputs["job"], inputs["filters"])
                
                # Parse response
                parsed = parse_response_optimized(response)
                return [link] + parsed
                
            except Exception as e:
                logger.error(f"Error processing {link}: {e}")
                return [link, f"❌ Error: {str(e)[:100]}"] + [""] * 17
        
        # Process files in batches
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.timeout),
            connector=aiohttp.TCPConnector(limit=50)
        ) as session:
            
            for i in range(0, len(links), config.batch_size):
                batch = links[i:i + config.batch_size]
                logger.info(f"Processing batch {i//config.batch_size + 1}/{(len(links)-1)//config.batch_size + 1}")
                
                # Process batch concurrently
                tasks = [process_file(session, link) for link in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter successful results
                for result in batch_results:
                    if isinstance(result, list):
                        results.append(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Task failed: {result}")
        
        logger.info(f"Successfully processed {len(results)} files, {len(expired)} expired")
        
        if not results:
            logger.warning("No results to process")
            return
        
        # Create and format sheet
        logger.info("Creating Google Sheet...")
        sheet = create_and_format_sheet(inputs["creds"], results)
        
        # Share sheet and send email
        sheet.share(inputs["email"], perm_type='user', role='writer')
        await send_email_async(inputs["email"], sheet.url, expired)
        
        processing_time = time.time() - start_time
        logger.info(f"🎯 Processing completed in {processing_time:.2f} seconds")

        asyncio.create_task(send_feedback_email_async(inputs["email"]))
        
    except Exception as e:
        logger.error(f"Fatal error in run_optimized: {e}")
        raise

# --- FASTAPI ENDPOINTS ---
@app.get("/")
def read_root():
    return {
        "message": "Optimized Resume Evaluation Service",
        "version": "2.0",
        "features": ["async processing", "concurrent downloads", "batch operations", "memory optimization"]
    }

@app.post("/")
async def process_form(request: Request, background_tasks: BackgroundTasks):
    """Enhanced endpoint with better error handling"""
    try:
        data = await request.json()
        logger.info(f"📥 Received request for: {data.get('Email address', 'unknown')}")
        
        # Validate required fields
        required_fields = [
            "Link to Google Drive Resumes Folder (Make sure it's public)",
            "Email address",
            "Job Description"
        ]
        
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return {
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }
        
        # Add background task
        background_tasks.add_task(run_optimized, data)
        
        return {
            "status": "processing",
            "email": data.get("Email address"),
            "message": "Resume evaluation started. You will receive an email when complete.",
            "estimated_time": "Processing time depends on the number of resumes"
        }
        
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        return {
            "status": "error",
            "message": "Failed to process request. Please check your input and try again."
        }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# --- CONFIGURATION ENDPOINT ---
@app.get("/config")
def get_config():
    """Get current processing configuration"""
    return {
        "max_concurrent_downloads": config.max_concurrent_downloads,
        "max_concurrent_evaluations": config.max_concurrent_evaluations,
        "batch_size": config.batch_size,
        "timeout": config.timeout
    }

@app.post("/config")
async def update_config(new_config: dict):
    """Update processing configuration"""
    global config
    for key, value in new_config.items():
        if hasattr(config, key) and isinstance(value, int) and value > 0:
            setattr(config, key, value)
    return {"status": "updated", "config": get_config()}