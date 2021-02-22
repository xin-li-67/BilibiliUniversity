import pika, json

params = pika.URLParameters('amqps://ymnjkdlq:FTozwvYgYfBTczkiABF6OOpJBhvhCRkR@beaver.rmq.cloudamqp.com/ymnjkdlq')

connection = pika.BlockingConnection(params)
channel = connection.channel()


def publish(method, body):
    properties = pika.BasicProperties(method)
    channel.basic_publish(exchange='', routing_key='admin', body=json.dumps(body), properties=properties)
