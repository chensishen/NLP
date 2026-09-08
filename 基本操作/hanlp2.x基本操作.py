import json

def t0():
    from hanlp_restful import HanLPClient
    from hanlp_common.document import  Document

    HanLP = HanLPClient('https://www.hanlp.com/api', auth=None, language='zh')   # auth不填则匿名，zh中文，mul多语种

    r = HanLP.parse(
        "2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。",
    )
    print(r)
    print(type(r))

    r = HanLP.semantic_textual_similarity([
        ('看图猜一电影名', '看图猜电影'),
        ('无线路由器怎么无线上网', '无线上网卡和无线路由器怎么用'),
        ('北京到上海的动车票', '上海到北京的动车票'),
    ])
    print(r)
    print(type(r))

def t1():
    import requests

    url = 'https://www.hanlp.com/api/parse'
    form = {
        'text': '2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。',
        'tokens': None,
        'tasks': 'tok',
        'skip_tasks': None,
        'language': 'zh'
    }
    print(json.dumps(form, ensure_ascii=False))
    headers = {
        # 'Authorization': f'Basic {_auth}'
    }
    response = requests.post(url, json=form, headers=headers)
    result = json.loads(response.text)
    print(result)





if __name__ == '__main__':
    # t0()
    t1()



