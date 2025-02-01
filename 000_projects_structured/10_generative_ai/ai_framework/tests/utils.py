from dotenv import load_dotenv
import warnings
import contextlib
import os
import requests
from urllib3.exceptions import InsecureRequestWarning


def get_mode(args):
    mode = "debug"

    if len(args) > 1:
        mode = args[1]

        if mode in ('test', 'prod'):
            load_dotenv()

    mode = mode.lower()
    return mode


requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

old_merge_environment_settings = requests.Session.merge_environment_settings


@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(
            self, url, proxies, stream, verify, cert
        )
        settings["verify"] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            pass
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except Exception as error:
                print(f"Error ssl controL: {error}")

def get_file_in_framework(semi_path_file):
    
    if semi_path_file.startswith("/"):
        semi_path_file = semi_path_file[1:]
    
    cwd = os.getcwd()
    return f"{cwd}/ai_framework/{semi_path_file}"