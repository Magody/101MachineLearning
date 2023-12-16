# relevant packages & modules
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import json
from tqdm import tqdm  # from tqdm.notebook
import os
from IPython.display import clear_output
import undetected_chromedriver as uc
import shutil
import pandas as pd
import signal
import sys
# appending custom lib
path_lib = os.environ["path_lib"]
sys.path.append(path_lib)
from Jobs import State, get_job_template, save_job, load_job, process_job_scrapper, signal_handler  # type: ignore

import os



import pandas as pd

class Item:
    
    attributes = []
    
    def set_attributes(self, dict_attributes:dict):
        self.attributes = []
        for key,value in dict_attributes.items():
            setattr(self, key, value)
            self.attributes.append(key)
    
    def get_serie(self, map_attributes_to_columns=dict()):
        if len(self.attributes) == 0:
            print("First add atributes with set_attributes")
            return pd.Series()
        
        if len(map_attributes_to_columns) > 0:
            # not supported
            return pd.Series()
        
        data = dict()
        for attribute in self.attributes:
            data[attribute] = getattr(self, attribute)
        
        return pd.Series(data=data)
            


class ScrapperBase:
    
    driver = None
    
    logged_in = False
    restarting = False
    
    cache = dict()
    
    stop = False
    
    def __init__(self, base_url, start=True, headless=False):
        
        self.base_url = base_url
        self.stop = False
        
        if start:
            self.driver = uc.Chrome(suppress_welcome=True)  # supress welcome of first installation
            self.driver.get(self.base_url)
            self.wait(1)
        
    def wait(self, t):
        time.sleep(t)
        
    def check_requisites(self):
        if not self.logged_in:
            print("Please login first")
            return False        
        return True
    
    
    def prewarming(self, time_wait=2):
        print("Prewarming not implemented")
        pass
    
    def login(self, username, password):
        self.logged_in = True
        print("Logged in not implemented. Setting to True")
        
    def obtain_urls_items(self, url_begin="google.com", early_stop=-1):
        print("OBTAIN_URLS_ITEMS not implemented.")
        if not self.check_requisites():
            return []
        
        queue = [url_begin]
        urls = []

        while len(queue) > 0:
            
            if early_stop != -1 :
                if len(urls) > early_stop:
                    break
                
            url_page = queue.pop()

            self.driver.get(url_page)
            self.wait(1)
            
            element_next = self.driver.find_element(By.XPATH, "//ul[@class='pagination']/li[last()]")
            if element_next.get_attribute("class") != "disabled":
                url_next = element_next.find_element(By.XPATH, "./a").get_attribute("href")
                queue.append(url_next)

        return urls
    
    def init_cache_items(self, df, urls_items:list):
        self.cache["urls"] = urls_items
        self.cache["index_seen"] = -1
        self.cache["urls_error"] = []
        self.cache["df"] = df
        self.cache["index_df"] = 0
        
    def set_cache_url_error(self, url:str, error:Exception):
        print("Error with basic parsing", url, error)
        self.cache["urls_error"].append(url)
        self.cache["index_seen"] += 1

    def update_cache(self, item_dict:dict):               
                
        item = Item()
        item.set_attributes(item_dict)
        # Warning: not parallelism/concurrency or RAM controled
        # TODO: this is too slow, improve
        self.cache["df"].at[self.cache["index_df"], :] = item.get_serie()
        self.cache["index_seen"] += 1
        self.cache["index_df"] += 1

    def scrape_items(self, timers: dict(),
                     url_begin="https://www.google.com", 
                     urls=None,
                     header=["title", "header2"], early_stop=-1):
        print("Scrape items not implemented")
        df = pd.DataFrame(columns=header)
        
        if not self.check_requisites():
            return df
        
        try:
            urls_items = urls
            if urls is None:
                print("URLs CAN'T BE NONE")
                return df
                
            
            self.init_cache_items(df, urls_items)
            
            timer_load_page = timers.get("timer_load_page", 2)
            timer_load_disqus = timers.get("timer_load_disqus",1)
            timer_load_reactions = timers.get("timer_load_reactions",1)
            
            
            for url in tqdm(urls_items):
                
                # initial
                self.driver.get(url)
                self.wait(timer_load_page)
                
                # EXTRACT THE ROW VARIABLES HERE: use self.extract_first/self.parse_multiple
                # on url error call: self.set_cache_url_error(url, error)
                
                item_dict = {
                    "title": "title",
                    "header2": "header2"
                }
                
                self.update_cache(item_dict)
                
                if self.stop:
                    return self.cache["df"], False
            print("Finished!")
            return self.cache["df"], True
        except Exception as error:
            print("General Error:", error)
            return self.cache["df"], False


    def extract_first(self, elements, mapping=None, default="none"):
        elements_len = len(elements)
        if elements_len == 0:
            return default
        
        
        try:
            element = elements[0].text
            if mapping is not None:
                element = mapping(element)
            return element
        except:
            return default
        
    def parse_multiple(self, web_elements, separator="|"):
                    
        out = ""
        for i in range(len(web_elements)):
            out += web_elements[i].text
            if i < len(web_elements)-1:
                out += separator        
        return out
                    
    def close(self):
        print("CLOSING OBJECT...")
        if self.driver:
            self.driver.close()
            
    def __del__(self):
        print("DESTRUCTING OBJECT")
        if not self.restarting:
            self.close()
            
            
class SeleniumApp:
    
    def __init__(self, scrapper: ScrapperBase, path_jobs:str):
        self.scrapper = scrapper
        self.path_jobs = path_jobs
        
    def run(self, username:str, password:str):
        self.scrapper.prewarming()
        if username is None and password is None:
            self.scrapper.logged_in = True
        else:
            self.scrapper.login(username, password)
    
    def create_jobs(self, url_begin, job_id_start=1, job_batch_items=100):
        """Run the scrapper with login before this.

        Args:
            url_begin (_type_): _description_
            job_batch_items (int, optional): _description_. Defaults to 100.
        """
                
        urls = self.scrapper.obtain_urls_items(url_begin=url_begin)
        items_total = len(urls)

        if os.path.isdir(self.path_jobs):
            shutil.rmtree(self.path_jobs)
        
        if not os.path.isdir(self.path_jobs):
            os.mkdir(self.path_jobs)        

        job_id = job_id_start
        i = 0
        while i < items_total:
            i_end = min(i+job_batch_items-1, items_total-1)
            
            items_batch = urls[i:i_end+1]
        
            job_name = f"{job_id}"
            
            folder_job = os.path.join(self.path_jobs, job_name)
            file_job = os.path.join(folder_job,"info.json")

            if os.path.isdir(folder_job):
                print("Already exists that job! First delete and then add. Creating side file... info0.json")
                file_job = file_job.replace("info", "info0")
            else:
                os.mkdir(folder_job)
            job = get_job_template(job_name)
            job["urls_queued"] = items_batch
            save_job(file_job, job)
            print(f"Generated job: {file_job}. With {len(items_batch)} urls_queued")
            
            job_id += 1
            i = i_end + 1
        
        try:
            with open(f"{self.path_jobs}/urls_backup.txt", "w+") as f_out:
                f_out.writelines(
                    map(lambda s: s+"\n", urls)
                )
        except Exception as e:
            print("Cant, create a backup", e)
        
        self.close()
    
    def close(self):
        self.scrapper.close()
        
    def work(self, timers:dict, job_name:str):
        """Run the scrapper with login before this.

        Args:
            timers (dict): _description_
            job_name (str): _description_
        """
        folder_job = os.path.join(self.path_jobs,job_name)
        file_job = os.path.join(folder_job,"info.json")

        if os.path.isdir(folder_job):
            job = load_job(file_job)

            df = pd.DataFrame()

            if job["state"] != State.END:
                process_job_scrapper(self.scrapper, file_job, job, timers)
            else:
                self.close()
                print("Job alredy ended!")

            
        else:
            print("That job doesn't exist")
            


def check_jobs_complete(path_jobs:str):
    
    if path_jobs.endswith("/"):
        path_jobs = path_jobs[:-1]
    
    jobs = os.listdir(path_jobs)
    jobs = [j for j in jobs if not j.endswith("txt")]
    
    jobs_incomplete = []
    for job in jobs:
        
        with open(f"{path_jobs}/{job}/info.json", "r") as f_in:
            job_info:dict = json.load(f_in)
            if job_info["state"] != State.END:
                jobs_incomplete.append(job)
    print(f"Incomplete:{jobs_incomplete}")


            
def merge_shards(path_shards:str, header:list, path_output:str = None)->pd.DataFrame:
    if path_shards.endswith("/"):
        path_shards = path_shards[:-1]
        
    if path_output is not None:
        if path_output.endswith("/"):
            path_output = path_output[:-1]
        
    shards = os.listdir(path_shards)
    print(f"Shards: {len(shards)}")

    df = pd.DataFrame(columns=header)

    for shard in shards:
        file_name = f"{path_shards}/{shard}"
        df_to_join = pd.read_csv(file_name)
        if len(df_to_join) == 0:
            print(f"Shard: {shard} is empty")
            continue
        df = pd.concat([df, df_to_join], axis=0, ignore_index=True)
        
    if path_output is not None:
        df.to_csv(f"{path_output}/merged_shards.csv",index=False)
    return df    
    
