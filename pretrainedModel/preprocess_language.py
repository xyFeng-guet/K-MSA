import requests
import random
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

    def BdTrans(self, textDataset):
        translateText = {
            'ch': [],
            'en': []
        }

        for text in textDataset:
            translateText['ch'].append(text)
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
            response = requests.post(self.url, params=data, headers=self.header)    # 发送post请求
            text = response.json()  # 返回的为json格式用json接收数据
            translateText['en'].append(text['trans_result'][0]['dst'])

        return translateText


def ReadSaveData(do, data):
    if do == 'read':
        with open('/opt/data/private/Project/Datasets/MSA_Datasets/SIMS/Processed/unaligned_39.pkl', 'rb') as f:
            data = pickle.load(f)

        rawText = []
        for types in ['train', 'valid', 'test']:
            for text in data[types]['raw_text']:
                rawText.append(text)
        return rawText
    else:
        with open('/opt/data/private/K-MSA/pretrained/pretrained_text.pkl', 'wb') as f:
            pickle.dump(data, f)
            f.close()


if __name__ == '__main__':
    appKey = '20240427002036678'            # 你在第一步申请的APP ID
    appSecret = 'q55okNouXDa0e8_qu95V'      # 公钥
    BaiduTranslate_test = BaiDuFanyi(appKey, appSecret)

    Text_data = ReadSaveData('read', None)

    Results = BaiduTranslate_test.BdTrans(Text_data)      # 要翻译的词组

    ReadSaveData('save', Results)
    print(Results)
