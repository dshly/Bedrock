import boto3
import json
import pprint

region = 'us-west-2'
# boto3_bedrock = boto3.client('bedrock',region)
bedrock_runtime = boto3.client('bedrock-runtime',region)

#sonnet#
model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
#Haiku#
#model_id = 'anthropic.claude-3-haiku-20240307-v1:0'

system_prompt = """你是一个帮助翻译剧本的助理。我会告诉你一些信息，你的任务是将用户提出的英文源文本翻译成中文。翻译时，请遵守以下规则：
            0, 不要改变初衷。
            1, 翻译前先了解上下文，保持语义连贯，阅读流畅，但不要故意夸大。
            2, 原文大多是对话式的，因此翻译仍应符合TikTok/短视频/视频博客/Youtube视频的上下文。注意避免使用通常不用于日常聊天的词语。
            3, 适当的时候保留一些专有名词或专业术语未翻译，注意前后一致。
            4, 在<result></result>中回复翻译。不要包含任何额外的内容。
            """

#user_contents = "We'll cover all of those things in a moment, but before we get started, this video doesn't have a sponsor, but it is supported by the thousands of you wonderful people who get value out of all of my courses, prints, presets and ebooks over at patk.com."
#messages = [{"role": "user", "content": user_contents}]

messages=[{ "role":'user', "content":[{'type':'text','text': "What is quantum mechanics? "}]},\
         { "role":'assistant', "content":[{'type':'text','text': "It is a branch of physics that \
         describes how matter and energy interact with discrete energy values "}]},\
         { "role":'user', "content":[{'type':'text','text': "Can you explain a bit more about discrete energies?"}]}]


max_tokens = 4096
def generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens):
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages
        }
    )

    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())

    return response_body

def generate_message_stream(bedrock_runtime, model_id, system_prompt, messages, max_tokens):
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages
        }
    )

    response = bedrock_runtime.invoke_model_with_response_stream(body=body, modelId=model_id)
    response_body = response.get('body')

    return response_body


#=====================Batch call and output========================#
#responses = generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens)
responses = generate_message(bedrock_runtime, model_id,system_prompt,messages,max_tokens=512,temp=0.5,top_p=0.9)
pprint.pprint(responses)
#=====================Batch call and output========================#



