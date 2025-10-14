import pandas as pd
import re
import os
import openai
from datetime import date
from datetime import date, timedelta
import requests
from datetime import date, timedelta, datetime
from slack_sdk import WebClient
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import time
from dotenv import load_dotenv
import streamlit as st

st.title("Clockify Automation Dashboard")

st.write("✅ The Clockify app is running successfully on Render!")

def split_message(text, max_len=60000):
    """Split long Slack messages into smaller chunks."""
    return [text[i:i + max_len] for i in range(0, len(text), max_len)]

load_dotenv()
openai.api_type = os.getenv("openai_api_type")
openai.api_version = os.getenv("openai_api_version")
openai.api_base = os.getenv("openai_api_base")
openai.api_key = os.getenv("openai_api_key")

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

API_KEY = os.getenv("CLOCKIFY_API_KEY")
WORKSPACE_ID = os.getenv("CLOCKIFY_WORKSPACE_ID")

slack_client = WebClient(token=SLACK_BOT_TOKEN)
app = App(token=SLACK_BOT_TOKEN)

print("Fetching data from Clockify")

def send_message_slack(channel, user_input, output):
    message = f"*Query:* {user_input}\n*Answer:* {output}"
    slack_client.chat_postMessage(channel=channel, text=message)

def sudo_download_file_command(channel_id):
    today = date.today()

    file_name = "clockify_full_report.csv"
    processing_message = app.client.chat_postMessage(
        channel=channel_id,
        text="💭 Processing your request... please wait."
    )
    app.client.chat_update(
        channel=channel_id,
        ts=processing_message["ts"],
        text="Please wait while data is being downloaded from Clockify. This may take a couple of minutes."
    )

    start_date = (today - timedelta(days=120)).isoformat() + "T00:00:00Z"
    end_date = today.isoformat() + "T23:59:59Z"

    url = f"https://reports.api.clockify.me/v1/workspaces/{WORKSPACE_ID}/reports/detailed"
    headers = {
        "X-Api-Key": API_KEY,
        "Content-Type": "application/json"
    }

    all_entries = []
    page = 1
    page_size = 100

    while True:
        payload = {
            "dateRangeStart": start_date,
            "dateRangeEnd": end_date,
            "sortOrder": "DESCENDING",
            "exportType": "JSON",
            "amountShown": "HIDE_AMOUNT",
            "detailedFilter": {
                "options": {"totals": "CALCULATE"},
                "page": page,
                "pageSize": page_size,
                "sortColumn": "ID"
            }
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e}")
            print(f"Response content: {response.text}")
            break
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            break

        data = response.json()
        entries = data.get("timeentries", data.get("timeEntries", []))
        if not entries:
            break

        all_entries.extend(entries)
        print(f"Page {page} fetched, {len(entries)} entries")
        page += 1

    if not all_entries:
        print("No data found.")
        return pd.DataFrame()

    formatted_rows = []
    for e in all_entries:
        start_iso = e['timeInterval']['start']
        end_iso = e['timeInterval']['end']

        start_date_str = datetime.fromisoformat(start_iso.rstrip('Z')).strftime("%m/%d/%Y") if start_iso else ""
        end_date_str = datetime.fromisoformat(end_iso.rstrip('Z')).strftime("%m/%d/%Y") if end_iso else ""

        if start_iso and end_iso:
            start_dt = datetime.fromisoformat(start_iso.rstrip('Z'))
            end_dt = datetime.fromisoformat(end_iso.rstrip('Z'))
            duration_hours = (end_dt - start_dt).total_seconds() / 3600
        else:
            duration_hours = 0

        tag_names = [tag['name'] for tag in e.get("tags", []) if 'name' in tag]

        formatted_rows.append({
            "Project": e.get("projectName", ""),
            "Client": e.get("clientName", ""),
            "Description": e.get("description", ""),
            "Task": e.get("taskName", ""),
            "User": e.get("userName", ""),
            "Tags": ", ".join(tag_names),
            "Start Date": start_date_str,
            "End Date": end_date_str,
            "Duration (decimal)": round(duration_hours, 2)
        })

    df = pd.DataFrame(formatted_rows)
    df.to_csv(file_name, index=False)
    print(f"All data saved to {file_name} ({len(df)} rows)")
    return df


def get_clockify_sheet(channel_id):
    file_name = "clockify_full_report.csv"
    today = date.today()

    # If file exists, read it
    if os.path.exists(file_name):
        print(f"{file_name} exists. Reading from file.")
        df = pd.read_csv(file_name)
        return df

    # Only download on Saturday
    if today.weekday() != 2:  # Not Saturday
        print("Clockify data can only be downloaded on Saturday. Returning empty DataFrame.")
        return pd.DataFrame()

    # Saturday: proceed to download
    processing_message = app.client.chat_postMessage(
        channel=channel_id,
        text="💭 Processing your request... please wait."
    )
    app.client.chat_update(
        channel=channel_id,
        ts=processing_message["ts"],
        text="Please wait while data is being downloaded from Clockify. This may take a couple of minutes."
    )

    start_date = (today - timedelta(days=120)).isoformat() + "T00:00:00Z"
    end_date = today.isoformat() + "T23:59:59Z"

    url = f"https://reports.api.clockify.me/v1/workspaces/{WORKSPACE_ID}/reports/detailed"
    headers = {
        "X-Api-Key": API_KEY,
        "Content-Type": "application/json"
    }

    all_entries = []
    page = 1
    page_size = 100

    while True:
        payload = {
            "dateRangeStart": start_date,
            "dateRangeEnd": end_date,
            "sortOrder": "DESCENDING",
            "exportType": "JSON",
            "amountShown": "HIDE_AMOUNT",
            "detailedFilter": {
                "options": {"totals": "CALCULATE"},
                "page": page,
                "pageSize": page_size,
                "sortColumn": "ID"
            }
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e}")
            print(f"Response content: {response.text}")
            break
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            break

        data = response.json()
        entries = data.get("timeentries", data.get("timeEntries", []))
        if not entries:
            break

        all_entries.extend(entries)
        print(f"Page {page} fetched, {len(entries)} entries")
        page += 1

    if not all_entries:
        print("No data found.")
        return pd.DataFrame()

    formatted_rows = []
    for e in all_entries:
        start_iso = e['timeInterval']['start']
        end_iso = e['timeInterval']['end']

        start_date_str = datetime.fromisoformat(start_iso.rstrip('Z')).strftime("%m/%d/%Y") if start_iso else ""
        end_date_str = datetime.fromisoformat(end_iso.rstrip('Z')).strftime("%m/%d/%Y") if end_iso else ""

        if start_iso and end_iso:
            start_dt = datetime.fromisoformat(start_iso.rstrip('Z'))
            end_dt = datetime.fromisoformat(end_iso.rstrip('Z'))
            duration_hours = (end_dt - start_dt).total_seconds() / 3600
        else:
            duration_hours = 0

        tag_names = [tag['name'] for tag in e.get("tags", []) if 'name' in tag]

        formatted_rows.append({
            "Project": e.get("projectName", ""),
            "Client": e.get("clientName", ""),
            "Description": e.get("description", ""),
            "Task": e.get("taskName", ""),
            "User": e.get("userName", ""),
            "Tags": ", ".join(tag_names),
            "Start Date": start_date_str,
            "End Date": end_date_str,
            "Duration (decimal)": round(duration_hours, 2)
        })

    df = pd.DataFrame(formatted_rows)
    df.to_csv(file_name, index=False)
    print(f"All data saved to {file_name} ({len(df)} rows)")
    return df

def gpt_response(input_str, table):
    # print(input_str)
    response = openai.ChatCompletion.create(
        engine="gpt-4o",
        messages= [
            {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
            {"role": "user", "content": f"""
            You are a data analysis assistant.
            Your task is to generate pandas code that answers the query **precisely** using the table.
            Following is the details about each column of table:
            - project : This column will give you the name of project.
            - client : This column will give you the name of client who own this project.
            - description : The project may have different tasks , this column will giive the more informaiton about each task.
            - task : This column will give you the detail of each task that are related to the project if the Task column is empty reffer to the column Description ot find details of the task.
            - user : This column will give you the information about developer name who is working on specific Task/Project.
            - tags : The project may have different type of tags like Billable(if the hours are within threshold), Unbillable( if the hours overshoot developer post their hour in this tag), Internal(if the work done by Devloper is Internal task and it is related to own company itself), Unapproved (if the hours are not approved by anyone, then they fall into this category).
            - start date : The date when specific task is started.It is in format mm-dd-yyyy.
            - end date : The date when specific task is finished.It is in format mm-dd-yyyy.
            - duration : Total number of hours that are used to complete specific Task
            
            Rules:
            - Remember every data of the table columns are in lowercase.
            - All computations related to dates should be done in dates not datetime
            - Carefully analyze all the columns that user is expecting as output.
            - Carefully analyze the user query and output that user is expecting.
            - Use 'df' as the dataframe variable.
            - Only use the columns that are defined above.
            - Approach and create the query step by step , do not do everything in one go.
            - If using Pandas in the query Always use import pandas as pd before using it in query .
            - If using any other module must import it in the top, because at the end i want to execute the code.
            - Only return the exact answer to the query, nothing more.
            - Do not generate summaries, explanations, or extra text.
            - If the query asks for information not present in the table, return exactly: "Sorry i can provide you the answer from the file only".
            - Be careful with date formats, column names, and data types.
            - Always store the final output in a variable called 'answer'.
            - Do not use any ascending or descending order in code
            - If user asks for query related to last week always look for last week from Sunday to Saturday from current date.
            - If user asks for query related to last Month always look for last Month from Currnet Month, must search by first calculating the dates and then applying query, but not time only date.
            - If user asks for query related to last Qaurter always find the last quarter months acccording to indian financial year system and create query according to these months,must search by first calculating the dates and then applying query, but not time only date.
            - Always include first and last date when querying with week,month or quarter.
            - Do all date computions in MM-DD-YYYY format only.
            - If user ask to generate query based on task ,description user or project name, always use contains keyword never look for exact match.
            - If user ask to generate query based on hour related to tag , always use exact match never look for contains.
            Input:
            Query: {input_str}
            Output:
            Generate python pandas code to answer this question using 'df' as the dataframe.
            Return code that computes the answer and stores it in a variable named 'answer'.
             """}
        ]
    )
    return response.choices[0].message.content

def summarizer(table, user_query):
    response = openai.ChatCompletion.create(
        engine="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
            {"role": "user", "content": f"""
            Your job is to summarize the answer of a query. 
            Summarize the answer **precisely** according to the user's query the user query was : {user_query}. 
            The answer is provided by LLM was : {table}. 
            if this response can be represented in table, return table starting and ending with ``` adjust the table width accordingly for symmetry in column widths. otherwise return simple string
            Do not add unnecessary details. 
            """}
        ]
    )
    return response.choices[0].message.content

# def load_data(channel_id): 
#     data = get_clockify_sheet(channel_id)
#     if not data.empty:
#     # Ensure all column names are strings first
#         data.columns = [str(col).lower() for col in data.columns]
#         data = data.applymap(lambda x: x.lower() if isinstance(x, str) else x)
#         if 'user' in data.columns:
#             data['user'] = data['user'].str.replace('.', ' ', regex=False)
#         if 'start date' in data.columns:
#             data['start date'] = data['start date'].str.replace('/', '-', regex=False)
#         if 'end date' in data.columns:
#             data['end date'] = data['end date'].str.replace('/', '-', regex=False)
#         if 'tags' in data.columns:
#             data['tags'] = data['tags'].str.replace('-', '', regex=False)
#         if 'project' in data.columns:
#             data['project'] = data['project'].str.replace('-', '', regex=False)
#         if 'description' in data.columns and 'task' in data.columns:
#             data['task'] = data.apply(
#                 lambda row: row['description'] if pd.isna(row['task']) or row['task'] == '' else row['task'],
#                 axis=1
#             )   
#         data.rename(columns={
#             'start date': 'start date',
#             'end date': 'end date',
#             'duration (decimal)': 'duration'
#         }, inplace=True)

#         for col in ['start date', 'end date']:
#             if col in data.columns:
#                 data[col] = pd.to_datetime(data[col], format="%m-%d-%Y", errors='coerce')
#         if 'duration' in data.columns:
#             data['duration'] = pd.to_numeric(data['duration'], errors='coerce')
#     return data
import pandas as pd

def load_data(channel_id):
    data = get_clockify_sheet(channel_id)
    
    if data.empty:
        return data 

    data.columns = [str(col).lower() for col in data.columns]

    data = data.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    if 'user' in data.columns:
        data['user'] = data['user'].str.replace('.', ' ', regex=False)
    if 'start date' in data.columns:
        data['start date'] = data['start date'].str.replace('/', '-', regex=False)
    if 'end date' in data.columns:
        data['end date'] = data['end date'].str.replace('/', '-', regex=False)
    if 'tags' in data.columns:
        data['tags'] = data['tags'].str.replace('-', '', regex=False)
    if 'project' in data.columns:
        data['project'] = data['project'].str.replace('-', '', regex=False)

    if 'description' in data.columns and 'task' in data.columns:
        data['task'] = data.apply(
            lambda row: row['description'] if pd.isna(row['task']) or row['task'] == '' else row['task'],
            axis=1
        )
    data.to_csv("clockify_full_report.csv", index=False)
    data.rename(columns={
        'duration (decimal)': 'duration'
    }, inplace=True)

    for col in ['start date', 'end date']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], format="%m-%d-%Y", errors='coerce')

    if 'duration' in data.columns:
        data['duration'] = pd.to_numeric(data['duration'], errors='coerce')

    return data


def clean_gpt_code(code: str) -> str:
    code = re.sub(r"```python", "", code, flags=re.IGNORECASE)
    code = re.sub(r"```", "", code, flags=re.IGNORECASE)
    code = code.strip()
    if "answer" not in code:
        code += "\nanswer = None"
        
    return code

@app.event("message")
def handle_message(message, say):
    user_text = message.get("text")
    prompt = user_text.lower()
    channel_id = message.get("channel")

    if user_text == "sudo downloadfiledatatilltoday":
        print("Downloading files")
        sudo_download_file_command(channel_id)
        user_text = ""

    data = load_data(channel_id)
    if data is not None:
        print("✅ File read successfully!")

        schema_info = "Table: Clockify\nColumns:\n"
        for col, dtype in zip(data.columns, data.dtypes):
            schema_info += f"- {col} ({dtype})\n"

        query_prompt = f"""
        Table Schema:
        {schema_info}
        
        Question:
        {prompt}
        """

        retries = 10
        delay = 2

        # ✅ Send initial "processing" message
        processing_message = app.client.chat_postMessage(
            channel=channel_id,
            text="💭 Processing your request... please wait."
        )

        for attempt in range(1, retries + 1):
            try:
                print(f"🔁 Attempt {attempt} of {retries}")
                raw_code = gpt_response(query_prompt, [])
                print(raw_code)

                pandas_code = clean_gpt_code(raw_code)
                local_vars = {"df": data}

                exec(pandas_code, {}, local_vars)
                answer = local_vars.get("answer", "No answer returned.")

                if isinstance(answer, pd.DataFrame):
                    print(answer)
                    table_str = answer.to_string(index=False)
                elif isinstance(answer, pd.Series):
                    print(answer.reset_index())
                    table_str = str(answer.to_dict())
                else:
                    table_str = str(answer)

                summarized = summarizer(table_str, prompt)
                summarized = summarized.capitalize()
                print(summarized)

                # ✅ Paginate long outputs
                chunks = split_message(summarized)
                if len(chunks) == 1:
                    app.client.chat_update(
                        channel=channel_id,
                        ts=processing_message["ts"],
                        text=chunks[0]
                    )
                else:
                    # Update first message
                    app.client.chat_update(
                        channel=channel_id,
                        ts=processing_message["ts"],
                        text=chunks[0]
                    )
                    # Post remaining chunks as follow-up messages
                    for chunk in chunks[1:]:
                        app.client.chat_postMessage(channel=channel_id, text=chunk)

                break

            except Exception as e:
                print(f"⚠️ Attempt {attempt} failed: {e}")
                print("Failed code:", raw_code)

                if attempt < retries:
                    if attempt > 2:
                        app.client.chat_update(
                            channel=channel_id,
                            ts=processing_message["ts"],
                            text=f"💭 Seems a complex query, taking a bit longer to compute..."
                        )
                    else:
                        app.client.chat_update(
                            channel=channel_id,
                            ts=processing_message["ts"],
                            text=f"💭 Processing your request... please wait."
                        )
                    print(f"⏳ Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    error_message = f"❌ Unable to connect to Server, Please re-submit the query"
                    app.client.chat_update(
                        channel=channel_id,
                        ts=processing_message["ts"],
                        text=error_message
                    )
                    send_message_slack(channel_id, user_text, error_message)
                    raise

if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()