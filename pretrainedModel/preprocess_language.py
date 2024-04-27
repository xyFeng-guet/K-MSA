import requests
import random
import json
import hashlib
import pickle


class BaiDuFanyi:
    def __init__(self, appKey, appSecret):
        self.url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
        self.appid = appKey
        self.secretKey = appSecret
        self.fromLang = 'zh'
        self.toLang = 'en'
        self.salt = random.randint(32768, 65536)
        self.header = {'Content-Type': 'application/x-www-form-urlencoded'}

    def BdTrans(self, text):
        sign = self.appid + text + str(self.salt) + self.secretKey
        md = hashlib.md5()
        md.update(sign.encode(encoding='utf-8'))
        sign = md.hexdigest()
        data = {
            "appid": self.appid,
            "q": text,
            "from": self.fromLang,
            "to": self.toLang,
            "salt": self.salt,
            "sign": sign
        }
        response = requests.post(self.url, params= data, headers= self.header)  # 发送post请求
        text = response.json()  # 返回的为json格式用json接收数据
        print(text)
        results = text['trans_result'][0]['dst']
        return results


def ReadData():
    with open('/opt/data/private/Project/Datasets/MSA_Datasets/SIMS/Processed/unaligned_39.pkl', 'rb') as f:
        data = pickle.load(f)

    rawText = []
    for types in ['train', 'valid', 'test']:
        for text in data[types]['raw_text']:
            rawText.append(text)

    return rawText


if __name__ == '__main__':
    appKey = '20240427002036678'            # 你在第一步申请的APP ID
    appSecret = 'q55okNouXDa0e8_qu95V'      # 公钥
    BaiduTranslate_test = BaiDuFanyi(appKey, appSecret)

    Text_data = ReadData()

    Results = BaiduTranslate_test.BdTrans(Text_data)      # 要翻译的词组
    print(Results)
