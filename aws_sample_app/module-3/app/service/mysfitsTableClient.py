import boto3
import json
import logging
import argparse

from collections import defaultdict

client = boto3.client('dynamodb')

def getMysfitsJson(list):
    mysfitsList = defaultdict(list)

    for item in items:
        mysfit = {}

        mysfit["mysfitId"] = item["MysfitId"]["S"]
        mysfit["name"] = item["Name"]["S"]
        mysfit["species"] = item["Species"]["S"]
        mysfit["description"] = item["Description"]["S"]
        mysfit["age"] = int(item["Age"]["N"])
        mysfit["goodevil"] = item["GoodEvil"]["S"]
        mysfit["lawchaos"] = item["LawChaos"]["S"]
        mysfit["thumbImageUri"] = item["ThumbImageUri"]["S"]
        mysfit["profileImageUri"] = item["ProfileImageUri"]["S"]
        mysfit["likes"] = item["Likes"]["N"]
        mysfit["adopted"] = item["Adopted"]["BOOL"]

        mysfitsList["mysfits"].append(mysfit)
    
    return mysfitsList

def getAllMysfits():
    response = client.scan(TableName='MysfitsTable')
    logging.info(response["Items"])
    mysfitList = getMysfitsJson(response["Items"])

    return json.dumps(mysfitList)

def queryMysfitItems(filter, value):
    response = client.query(
        TableName='MysfitsTable',
        IndexName=filter+'Index',
        KeyConditions={
            filter: {
                'AttributeValueList': [
                    {
                        'S': value
                    }
                ],
                'ComparisonOperator': "EQ"
            }
        }
    )

    mysfitList = getMysfitsJson(response["Items"])
    
    return json.dumps(mysfitList)

def queryMysfits(queryParam):
    logging.info(json.dumps(queryParam))

    filter = queryParam['filter']
    value = queryParam['value']

    return queryMysfitItems(filter, value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filter')
    parser.add_argument('-v', '--value')
    args = parser.parse_args()

    filter = args.filter
    value = args.value

    if args.filter and args.value:
        print(f'filter is {args.filter}')
        print(f'value is {args.value}')
        print('Getting filtered values')

        items = queryMysfitItems(args.filter, args.value)
    else:
        print("Getting all values")
        items = getAllMysfits()

    print(items)