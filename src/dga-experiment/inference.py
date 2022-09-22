import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data = 'sites.google.com'
data = 'stackoverflow.com'
data = 'zuugmhzwx.biz'
data = 'sokgosiioakymime.org'
# data = 'lsk8wo1196v3o1mwilk41t6fg2t.com'
# data = 'havelegislaturesthethe.com'

body = str.encode(json.dumps(data))

file1 = open('dga-experiment/keys.txt', 'r')
Lines = file1.readlines()
url = Lines[0].split(',')[1]
api_key = Lines[1].split(',')[1]

print(url)
print(api_key)

# The azureml-model-deployment header will force the request to go to a specific deployment.
# Remove this header to have the request observe the endpoint traffic rules
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print("URL:", data," Probabilities (Clean, DGA)",result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))