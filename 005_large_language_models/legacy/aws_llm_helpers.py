def get_dummy_event_context(queryStringParameters):
    event = {
        "version": "1.0",
        "resource": "/ciiuPredictor",
        "path": "/default/ciiuPredictor",
        "httpMethod": "GET",
        "headers": {
            "Content-Length": "0",
            "Host": "jszpvnqtu6.execute-api.us-east-1.amazonaws.com",
            "Postman-Token": "c8055f43-bb01-4258-8a96-93a4015f2b92",
            "User-Agent": "PostmanRuntime/7.36.0",
            "X-Amzn-Trace-Id": "Root=1-65957a73-140a703618b0f82f7e9ac521",
            "X-Forwarded-For": "163.116.226.80",
            "X-Forwarded-Port": "443",
            "X-Forwarded-Proto": "https",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br",
            "cache-control": "no-cache",
        },
        "multiValueHeaders": {
            "Content-Length": ["0"],
            "Host": ["jszpvnqtu6.execute-api.us-east-1.amazonaws.com"],
            "Postman-Token": ["c8055f43-bb01-4258-8a96-93a4015f2b92"],
            "User-Agent": ["PostmanRuntime/7.36.0"],
            "X-Amzn-Trace-Id": ["Root=1-65957a73-140a703618b0f82f7e9ac521"],
            "X-Forwarded-For": ["163.116.226.80"],
            "X-Forwarded-Port": ["443"],
            "X-Forwarded-Proto": ["https"],
            "accept": ["*/*"],
            "accept-encoding": ["gzip, deflate, br"],
            "cache-control": ["no-cache"],
        },
        "queryStringParameters": queryStringParameters,
        "multiValueQueryStringParameters": {"query": ["comercio"], "version": ["2"]},
        "requestContext": {
            "accountId": "108310387064",
            "apiId": "jszpvnqtu6",
            "domainName": "jszpvnqtu6.execute-api.us-east-1.amazonaws.com",
            "domainPrefix": "jszpvnqtu6",
            "extendedRequestId": "Q-ASGjqeIAMEVJw=",
            "httpMethod": "GET",
            "identity": {
                "accessKey": None,
                "accountId": None,
                "caller": None,
                "cognitoAmr": None,
                "cognitoAuthenticationProvider": None,
                "cognitoAuthenticationType": None,
                "cognitoIdentityId": None,
                "cognitoIdentityPoolId": None,
                "principalOrgId": None,
                "sourceIp": "163.116.226.80",
                "user": None,
                "userAgent": "PostmanRuntime/7.36.0",
                "userArn": None,
            },
            "path": "/default/ciiuPredictor",
            "protocol": "HTTP/1.1",
            "requestId": "Q-ASGjqeIAMEVJw=",
            "requestTime": "03/Jan/2024:15:17:07 +0000",
            "requestTimeEpoch": 1704295027540,
            "resourceId": "ANY /ciiuPredictor",
            "resourcePath": "/ciiuPredictor",
            "stage": "default",
        },
        "pathParameters": None,
        "stageVariables": None,
        "body": None,
        "isBase64Encoded": False,
    }
    context = {}
    return event, context

import warnings
import contextlib

import requests
from urllib3.exceptions import InsecureRequestWarning
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

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

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
            except:
                pass

def generate_vectors(embedder, df, column_name_content):
    df = embedder.split_and_explode(
        df, column_name_content_ciiu=column_name_content
    ).reset_index()

    df = embedder.embed_dataframe(
        df,
        column_name_content,
        check_cost=True
    )
    return df