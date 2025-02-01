# Setup and login
import json
import os
import sys
import signal


class State:
    BEGIN = 0
    PENDING_URLS = 1
    PENDING_ITEMS = 2
    END = 3
    PENDING_URLS_ITEMS = 4

def get_job_template(job_name):
    return {
            "state": State.BEGIN,
            "name": job_name,
            "urls_queued": [],
            "urls_error": [],
            "execution_time": 0
        }

def save_job(file_job, job):
    with open(file_job, 'w') as f:
        json.dump(job, f, ensure_ascii=False, indent=4)
        
def load_job(file_job):
    with open(file_job) as f:
        return json.load(f)
    
    
def close_job_scrapper(is_completed, df):
    """Close the job. Requires global vars

    Args:
        is_completed (bool): _description_
        df (DataFrame): _description_
    """
    global scrapper_reference
    global file_job, job 
    print(f"Stopping scraper...Completed: {is_completed} DF type: {type(df)}")
    scrapper_reference.stop = True
    print("Saving job...")
    
    pending = []
    if not is_completed:
        index_seen = scrapper_reference.cache.get("index_seen",-1)
        pending = scrapper_reference.cache.get("urls",[])
        if len(pending) > 0:
            pending = pending[index_seen+1:]
          
        df = scrapper_reference.cache.get("df", None)
        
    job['urls_queued'] = pending
    
    job['urls_error'] = scrapper_reference.cache.get("urls_error", [])
    save_job(file_job, job)
    
    if df is not None:
        folder_shards = os.path.join("shards")
        if not os.path.isdir(folder_shards):
            os.mkdir(folder_shards)
            
        df.to_csv(f"{folder_shards}/db_job-{job['name']}_items-run-{job['execution_time']}.csv", index=False)
    
    print(f"Closing with: {len(scrapper_reference.cache.get('urls',[])) -len(job['urls_queued'])} completed, {len(job['urls_queued'])} pending urls and {len(job['urls_error'])} error urls")
    

def signal_handler(sig, frame):
    close_job_scrapper(False, None)
    
    try:
        sys.exit()
    except:
        print("Problems with sys.exit!")
        sys.exit()
        
def process_job_scrapper(scrapper, fj, j, timers: dict):
    """Process_job in every item. The scrapper must be logged in if needed. And instance
    should have been open.

    Args:
        scrapper (_type_): _description_
        job (_type_): _description_
        timers (dict): _description_
    """
    global scrapper_reference
    global file_job, job 
    file_job = fj
    job = j
    
    scrapper_reference = scrapper
    signal.signal(signal.SIGINT, signal_handler)

    state = job["state"]
    if state == State.BEGIN or state == State.PENDING_ITEMS:
        scrapper.stop = False
        urls_items = job["urls_queued"]
        urls_items.extend(job["urls_error"])  # try again the errors
        job["urls_error"] = []
        job["execution_time"] += 1
        
        print(f"Pending work...{len(urls_items)} urls in queue")
        
        df, completed = scrapper.scrape_items(timers, urls=urls_items)
        
        print("scrape_items ended")
        
        if completed and len(scrapper.cache["urls_error"]) == 0: 
            job["state"] = State.END
        
        close_job_scrapper(completed, df)
    elif state == State.END:
        print("Work already completed!")
    else:
        print(f"Can't handle this state: {state}")