from __future__ import print_function

import base64
import json
import requests

def retrieveMysfit(mysfitId):
    apiEndpoint = 'REPLACE_ME_API_ENDPOINT' + '/mysfits/' + str(mysfitId)
    mysfit = requests.get(apiEndpoint).json()

    return mysfit

def processRecord(event, context):
    output = []

    for record in event['records']:
        print('Processing record: ' + record['recordId'])

        # kinesis firehose expects record payloads to be sent as encoded strings,
        # so we must decode the data first to retrieve the click record.
        click = json.loads(base64.b64decode(record['recordId']))
        mysfitId = click['mysfitId']
        mysfit = retrieveMysfit(mysfitId)

        enrichedClick = {
                'userId': click['userId'],
                'mysfitId': mysfitId,
                'goodevil': mysfit['goodevil'],
                'lawchaos': mysfit['lawchaos'],
                'species': mysfit['species']
            }
        
        # create the output record that Kinesis Firehose will store in S3.
        output_record = {
            'recordId': record['recordId'],
            'result': 'Ok',
            'data': base64.b64encode(json.dumps(enrichedClick).encode('utf-8') + b'\n').decode('utf-8')
        }
        output.append(output_record)
    
    print('Successfully processed {} records.'.format(len(event['records'])))

    # return the enriched records to Kiesis Firehose.
    return {'records': output}