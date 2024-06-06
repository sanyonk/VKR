import re
import spacy
import requests
import streamlit as st
from newspaper import Article
from transformers import pipeline
from collections import OrderedDict

nlp = spacy.load("en_core_web_sm")

def fetch_news_text(url):
    article = Article(url)
    article.download()
    article.parse()
    text = article.text
    cleaned_text = text.replace('\n', ' ')
    cleaned_text = cleaned_text.replace('  ', ' ')
    return cleaned_text

def cve_format(text):
    cve_pattern = r'\bCVE-\d{4}-\d{4,}\b'
    cve_numbers = re.findall(cve_pattern, text)
    return cve_numbers

def get_malware(text):
    ner = pipeline("ner", model = 'AI4Sec/cyner-xlm-roberta-base', aggregation_strategy ='average')
    ent = ner(text)
    malware_entities = [(item['word']) for item in ent if item['entity_group'] == 'Malware']
    entities = []
    for entity in malware_entities:
        entity = entity.replace(',', '').replace('"', '').replace(')', '').replace('.', '')
        entities.append(entity)
    non_duplicates = []
    for vpo in list(set(entities)):
        non_duplicates.append(vpo)
    return non_duplicates
    
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', multi_label = True)

def get_monetary_damage(text):
    monetary_damage = []
    doc = nlp(text)
    sentences = text.split(".")
    indices = []
    for i, sentence in enumerate(sentences):
        if any(ent.label_ == 'MONEY' for ent in nlp(sentence).ents):
            indices.append(i)
    
    for index in indices:
        sentence = sentences[index].strip()
        result = classifier(sentence, ['damage'])
        if result['scores'][0] > 0.5:
            # print(sentence)
            for ent in nlp(sentence).ents:
                if ent.label_ == 'MONEY':
                    monetary_damage.append(ent.text)
    return monetary_damage

def get_victim(text):
    api_token = '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'
    url_question_pipe = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    questions = ['Who lost money?', 'Who was attacked in the attack?']
    victim = []

    for question in questions:
        payload = {"inputs": {"question": question, "context": text}}
        headers = {'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'}
        response = requests.post(url_question_pipe, json=payload, headers=headers)
        result = response.json()
        victim.append(result.get('answer'))
    
    non_duplicates = []

    for i in list(set(victim)):
        non_duplicates.append(i)
    
    return non_duplicates

def get_attacker(text):
    api_token = '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'
    url_question_pipe = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    questions = ['Who carried out the attack?', 'What malware was used for the attack?']
    attacker = []

    for question in questions:
        payload = {"inputs": {"question": question, "context": text}}
        headers = {'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'}
        response = requests.post(url_question_pipe, json=payload, headers=headers)
        result = response.json()
        attacker.append(result.get('answer'))
    
    return attacker

def get_date(text):
    api_token = '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'
    url_question_pipe = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    questions = ['What was the date of the attack?', 'when did the event happen?']
    date = []

    for question in questions:
        payload = {"inputs": {"question": question, "context": text}}
        headers = {'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'}
        response = requests.post(url_question_pipe, json=payload, headers=headers)
        result = response.json()
        date.append(result.get('answer'))
        
    dupl = []
    for dates in list(set(date)):
        dupl.append(dates)
    
    return dupl

def get_consequences(text):
    api_token = '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'
    url_zero_shot = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    payload = {"inputs": text, "parameters": {"candidate_labels": ["Data Breaches", "Financial Loss",
                                                                   "Reputational Damage","Downtime and Operational Disruption",
                                                                   "Legal and Regulatory Consequences",
                                                                   "Loss of Intellectual Property", "Identity Theft",
                                                                   "Fraudulent Activities", "Damage to Customer Trust and Relationships",
                                                                   "Potential lawsuits or fines"], "multi_label": True}}

    headers = {'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'}
    response = requests.post(url_zero_shot, json=payload, headers=headers)
    result = response.json()
    max_label = max(result["scores"])
    max_label_index = result["scores"].index(max_label)
    
    return result['labels'][max_label_index]

def description_of_malware(text):
    api_token = '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'
    url_text_generation = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    malwares = get_malware(text)
    descriptions = []
    for malware in malwares:
        payload = {"inputs": f"What {malware} malware is?"}
        headers = {'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'}
        response = requests.post(url_text_generation, json=payload, headers=headers)
        result = response.json()
        result = result[0]['generated_text'].replace(f'What {malware} malware is?', '').strip().replace('\n', ' ').replace('  ', ' ')
        sentences = re.split(r'(?<=[.!?])\s+', result)
        descriptions.append(' '.join(sentences[:-1]))
        
    return descriptions

def get_target_mass_attack(text):
    api_token = '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'
    url_zero_shot = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    payload = {"inputs": text, "parameters": {"candidate_labels": ["Targeted attack", "Mass attack"], "multi_label": True}}
    headers = {'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'}
    response = requests.post(url_zero_shot, json=payload, headers=headers)
    result = response.json()
    max_label = max(result["scores"])
    max_label_index = result["scores"].index(max_label)
    
    return result['labels'][max_label_index]

def get_method_of_attack(text):
    api_token = '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'
    url_zero_shot = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    methods = ['Use of malware', 'Compromise of credentials',
               'Social engineering', 'Exploitation of vulnerabilities',
               'Use of legal software',
               'Compromise of the supply chain or trusted communication channels',
               'DDoS', 'Other']
    
    payload = {"inputs": text, "parameters": {"candidate_labels": methods, "multi_label": True}}
    headers = {'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'}
    response = requests.post(url_zero_shot, json=payload, headers=headers)
    result = response.json()
    max_label = max(result["scores"])
    max_label_index = result["scores"].index(max_label)
    
    return result['labels'][max_label_index]


def get_category_of_victim(text):
    api_token = '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'
    url_zero_shot = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

    labels = [['Government institutions', 'Industry', 'Medical institutions',
               'Transport', 'Media', 'Telecommunications', 'Without reference to the industry',
               'IT companies', 'Individuals', 'Science and education'], 
               ['Defense enterprises', 'Financial organizations', 'Others',
                'Trade', 'Service sector', 'Blockchain projects', 'Online services']]

    dictionaries = []
    
    for victim in text:
        for label in labels:
            payload = {"inputs": victim, "parameters": {"candidate_labels": label, "multi_label": True}}
            headers = {'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'}
            response = requests.post(url_zero_shot, json=payload, headers=headers)
            result = response.json()
            max_label = max(result["scores"])
            max_label_index = result["scores"].index(max_label)
            dictionaries.append(dict(label=result["labels"][max_label_index], score=max_label))
             
    if len(text) == 1:
        max_label = max(dictionaries, key=lambda x: x['score'])['label']
        
        return max_label
    else:
        first_half = dictionaries[:len(dictionaries)//len(text)]
        second_half = dictionaries[len(dictionaries)//len(text):]
        max_label = max(first_half, key=lambda x: x['score'])['label']
        max_label_second = max(second_half, key=lambda x: x['score'])['label']
        
        return f'{max_label}, {max_label_second}'
        
type_vpo = {'Conti': 'Шифровальщик', 'DarkCrystal RAT': 'ВПО для удаленного управления', 
            'LockBit': 'Шифровальщик', 'Hive': 'Шифровальщик', 'Quantum': 'Загрузчик', 
            'AstraLocker': 'Шифровальщик', 'Avos Locker': 'Шифровальщик', 'OrBit': 'Шпионское ПО', 
            'XMRig': 'ВПО для удаленного управления', 'HavanaCrypt': 'Шифровальщик', 
            'Rozena': 'ВПО для удаленного управления', 'LockBit 3.0': 'Шифровальщик', 
            'Meterpreter, AnchorMail, CobaltStrike, IcedID, Forest, Bumblebee': 'ВПО для удаленного управления', 
            'Bisonal, Royal Road': 'ВПО для удаленного управления', 'Vice Society': 'Шифровальщик', 
            'RedAlert': 'Шифровальщик', 'ZxxZ': 'ВПО для удаленного управления', 
            'Brute Ratel': 'ВПО для удаленного управления', 'Vsingle': 'Загрузчик', 
            'IcedID': 'Банковский троян', 'Luca Stealer': 'Банковский троян', 'GammaLoad': 'Загрузчик', 
            'Predator': 'Шпионское ПО', 'Subzero': 'Загрузчик', 'Gootkit, Cobalt Strike': 'ВПО для удаленного управления', 
            'Lofy Stealer': 'Банковский троян', 'Eking (Phobos)': 'Шифровальщик', 'ABCsoup': 'Рекламное ПО', 
            'Colibri, Vidar': 'Шпионское ПО', 'BarbDownloader,BarbWire Backdoor, VolatileVenom': 'ВПО для удаленного управления', 
            'FFDroid': 'Шпионское ПО', 'NetSupport RAT': 'ВПО для удаленного управления', 'Denonia': 'Майнер', 
            'HilalRAT': 'ВПО для удаленного управления', 'SharkBot': 'Банковский троян', 'BlackCat, ExMatter, Mimikatz': 'Шифровальщик',
            'LockBit 2.0': 'Шифровальщик', 'Mirai': 'ВПО для удаленного управления', 'SolarMaker': 'ВПО для удаленного управления', 
            'NB65': 'Шифровальщик', 'Octo': 'Банковский троян', 'META Stealer': 'Шпионское ПО', 
            'Cyclops Blink': 'ВПО для удаленного управления', 'FakeCalls': 'Банковский троян', 'Qbot': 'Загрузчик', 
            'ALPHV': 'Шифровальщик', 'Industroyer 2.0, CADDYWIPER': 'Другой', 
            'LockBit, LaZagne, Mimikatz,': 'ВПО для удаленного управления', 'Enemybot': 'ВПО для удаленного управления', 
            'Linux Tsunami, майнеры': 'Майнер', 'Crazyfia': 'Загрузчик', 'ZingoStealer, XMRig': 'Шпионское ПО', 
            'CVE-2022-0609': 'Загрузчик', 'Cobalt Strike, Trickbot,': 'ВПО для удаленного управления', 
            'InnoStealer': 'Шпионское ПО', 'Hive, Cobalt Strike': 'ВПО для удаленного управления', 
            'Hammer, DDoS-Ripper, Wiper': 'ВПО для удаленного управления', 'Revil': 'Шифровальщик', 
            'Lemon_Duck, XMRig': 'Майнер', 'IcedID, Quantum': 'Шифровальщик', 'Emotet, Cobalt Strike': 'Загрузчик', 
            'Cobalt Strike, Metasploit': 'Загрузчик', 'Stormous': 'Шифровальщик', 'Black Basta': 'ВПО для удаленного управления',
            'Emotet': 'ВПО для удаленного управления', 'Onyx': 'Шифровальщик', 'Bumblebee': 'Загрузчик', 'BlackBasta': 'Загрузчик',
            'QUIETEXIT': 'ВПО для удаленного управления', 'Shadowpad, PlugX, Gunters': 'ВПО для удаленного управления', 
            'PlugX': 'ВПО для удаленного управления', 'NetDooka': 'ВПО для удаленного управления', 'Jester Stealer': 'Шпионское ПО', 
            'Cuba': 'Шифровальщик', 'Joker': 'Шпионское ПО', 'MobOk': 'Банковский троян', 'Vesub': 'Банковский троян', 
            'GriftHorse.I.': 'Банковский троян', 'Nerbian RAT': 'ВПО для удаленного управления', 'RedLine': 'Загрузчик', 
            'Vidar': 'Шпионское ПО', 'AveMariaRAT, BitRAT, PandoraHVNC': 'ВПО для удаленного управления', 'Sysrv, XMRig': 'Майнер', 
            'Sysrv': 'Майнер', 'PowerShell RAT': 'ВПО для удаленного управления', 'Spectra': 'Загрузчик', 
            'Facestealer': 'Шпионское ПО', 'MyKLoadClient, Zupdax,  Downloader.Climax, RtlShare, PlugX, Deed RAT, BH_A006': 'Загрузчик',
            'DeadBolt': 'Шифровальщик', 'NukeSped, Jin Miner': 'Майнер', 'XOR DDoS': 'ВПО для удаленного управления', 
            'Cl0p': 'Шифровальщик', 'INDUSTROYER2, CADDYWIPER, ORCSHRED, SOLOSHRED, AWFULSHRED': 'ПО, удаляющее данные', 
            'Cobalt Strike': 'Шифровальщик', 'Cheers': 'Шифровальщик', 'BlackCat AKA ALPHV': 'Шифровальщик', 'Xloader': 'Шпионское ПО',
            'CoolRAT': 'ВПО для удаленного управления', 'GuLoader, RemcosRAT': 'ВПО для удаленного управления', 'GoodWill': 'Шифровальщик',
            'SVCReady': 'ВПО для удаленного управления', 'Cuba, BUGHATCH': 'Шифровальщик', 'Cobalt Strike, XMRig': 'Майнер', 
            'RansomHouse': 'Шифровальщик', 'eCh0raix': 'Шифровальщик', 'Hermit': 'Шпионское ПО', 'Meteor': 'ПО, удаляющее данные',
            'YTStealer': 'Шпионское ПО', 'Yanluowang': 'Шифровальщик', 'More_Eggs': 'ВПО для удаленного управления', 
            'BlackCat': 'Шифровальщик', 'Scieron, HeaderTip': 'ВПО для удаленного управления', 'Black Basta, Gh0st RAT': 'Шифровальщик',
            'TRITON (TRISIS)': 'ВПО для удаленного управления', 'CopperStealer, Vidar': 'Шпионское ПО', 'Panchan, XMRig, nbhash': 'Майнер',
            'Matanbuchus, Cobalt Strike': 'Загрузчик', 'SMSFactory': 'Другой', 'CreepyDrive, CreepySnail': 'ВПО для удаленного управления',
            'Ousaban': 'Банковский троян', 'X-FILES': 'Шпионское ПО', 'Ragnar Locker': 'Шифровальщик', 'CL0P': 'Шифровальщик', 
            'Dracarys': 'Шпионское ПО', 'LazaSpy': 'Шпионское ПО', 'Industrial Spy': 'Шифровальщик', 
            'China Chopper': 'ВПО для удаленного управления', 'LV': 'Шифровальщик', 'FakeCrack': 'Шпионское ПО', 
            'AsyncRAT': 'ВПО для удаленного управления', 'WannaFriendMe': 'Шифровальщик', 'CrescentImp': 'Рекламное ПО', 
            'Cerber2021': 'Шифровальщик', 'AvosLocker': 'Шифровальщик', 'Hello XD': 'Шифровальщик', 'Night Sky': 'Шифровальщик', 
            'XMRig, Cobalt Strike': 'Майнер', 'MaliBot': 'Банковский троян', 'Daixin Team': 'Шифровальщик', 'SHARPEXT': 'Шпионское ПО',
            'FakeUpdates, Raspberry Robin, Cobalt Strike': 'Загрузчик', 'HiddenAds': 'Рекламное ПО', 'BianLian': 'Шифровальщик', 
            'Qakbot': 'Банковский троян', 'Lilith': 'Шифровальщик', 'WarzoneRAT': 'ВПО для удаленного управления', 
            'AysncRAT, LimeRAT': 'ВПО для удаленного управления', 'Revive': 'Банковский троян', 
            'Cobalt Strike, ShadowPad': 'ВПО для удаленного управления', 'CredoMap': 'Шпионское ПО', 
            'CrimsonRAT, ObliqueRAT, CapraRAT': 'ВПО для удаленного управления', 'Sality': 'ВПО для удаленного управления', 
            'Chinoxy': 'Загрузчик', 'WatchDog': 'Майнер', 'Nimbda, Yahoyah, Tclient': 'ВПО для удаленного управления', 
            'Keona Clipper': 'Банковский троян', 'RelicRace, RelicSource, Formbook, Snake Keylogger': 'Шпионское ПО', 
            'CosmicStrand': 'Другой', 'Racoon Stealer': 'Шпионское ПО', 'Karakurt': 'Шифровальщик', 'Ljl Backdoor': 'Шпионское ПО',
            'CharmPower, PINEFLOWER': 'Шпионское ПО', 'WoodyRAT': 'ВПО для удаленного управления', 'GwisinLocker': 'Шифровальщик',
            'BlackByte': 'Шифровальщик', 'Facesteaer': 'Шпионское ПО', 'Coper': 'Шпионское ПО', 'CloudMensis': 'Шпионское ПО', 
            'Grimplant, Graphsteel': 'ВПО для удаленного управления', 
            'Grimplant, Graphsteel, Remote Utilities': 'ВПО для удаленного управления', 'Luna': 'Шифровальщик', 
            'KonniRAT': 'ВПО для удаленного управления', 'GoMet': 'ВПО для удаленного управления', 
            'Manjusaka, Cobalt Strike': 'ВПО для удаленного управления', 'Amadey, SmokeLoader': 'Загрузчик', 
            'DevilsTongue': 'Шпионское ПО', 'EvilNum': 'Загрузчик', 'SolidBit': 'Шифровальщик', 'BRATA': 'ВПО для удаленного управления',
            'Bumblebee, CobaltStrike, Diavol, Conti': 'ВПО для удаленного управления', 'Orchard': 'ВПО для удаленного управления', 
            'Donut Leaks': 'Шифровальщик', 'SmokeLoader': 'Загрузчик', 'Cuba, ROMCOM RAT': 'ВПО для удаленного управления', 
            'RansomEXX': 'Шифровальщик', 'SRG': 'Шифровальщик', 'SOVA': 'Банковский троян', 'BROWSER STEALER': 'Шпионское ПО', 
            'Everest': 'Шифровальщик'}

def get_type_vpo(non_duplicates):
    
    type = []
    for key, value in type_vpo.items():
        for malware in non_duplicates:
            if malware in key:
                type.append(f"{malware}: {value}")
                
    drop_duplicates = []
    for vpo in list(set(type)):
        drop_duplicates.append(vpo)
        
    return drop_duplicates

group_dict = {'Conti': 'Conti', 'DarkCrystal RAT': 'UAC-0113', 'LockBit': 'LockBit', 'Hive': 'Hive', 'Quantum': 'Quantum', 
              'AstraLocker': 'AstraLocker', 'Avos Locker': 'Avos Locker', 'OrBit': 'OrBit', 'XMRig': 'WatchDog', 
              'HavanaCrypt': 'HavanaCrypt', 'LockBit 3.0': 'LockBit', 
              'Meterpreter, AnchorMail, CobaltStrike, IcedID, Forest, Bumblebee': 'TrickBot', 'Bisonal, Royal Road': 'Tonto', 
              'Vice Society': 'Vice Society', 'RedAlert': 'RedAlert', 'ZxxZ': 'Bitter', 'Brute Ratel': 'APT29 AKA Cosy Bear', 
              'Vsingle': 'Lazarus Group', 'IcedID': 'TA578', 'Luca Stealer': 'Luca Stealer', 'GammaLoad': 'Armageddon', 
              'Subzero': 'Knotweed', 'Eking (Phobos)': 'Eking', 'BarbDownloader,BarbWire Backdoor, VolatileVenom': 'APT-C-23', 
              'FFDroid': 'FFDroider', 'NetSupport RAT': 'Parrot TDS', 'HilalRAT': 'UNC788', 'SharkBot': 'SharkBot', 
              'BlackCat, ExMatter, Mimikatz': 'BlackCat AKA ALPHV', 'LockBit 2.0': 'LockBit', 'Mirai': 'Mirai', 
              'SolarMaker': 'SolarMaker', 'NB65': 'NB65', 'Octo': 'Octo', 'META Stealer': 'META', 'Cyclops Blink': 'Sandworm', 
              'Qbot': 'ТА570', 'ALPHV': 'ALPHV aka BlackCat', 'Industroyer 2.0, CADDYWIPER': 'Sandworm', 
              'LockBit, LaZagne, Mimikatz,': 'LockBit', 'Enemybot': 'Keksec', 
              'Linux Tsunami, майнеры': 'Несколько субъектов угроз', 'Crazyfia': 'Fodcha', 'ZingoStealer, XMRig': 'Haskers Gang', 
              'CVE-2022-0609': 'APT38 AKA Lazarus Group', 'Cobalt Strike, Trickbot,': 'Trickbot (Conti)', 
              'Hive, Cobalt Strike': 'Hive', 'Hammer, DDoS-Ripper, Wiper': 'T3 Dimension Team', 'Revil': 'Revil', 
              'Lemon_Duck, XMRig': 'Lemon_Duck', 'IcedID, Quantum': 'Quantum', 'Emotet, Cobalt Strike': 'Emotet', 
              'Cobalt Strike, Metasploit': 'Rocket Kitten и еще несколько группировок', 'Stormous': 'Stormous', 
              'Black Basta': 'Black Basta', 'Emotet': 'Emotet', 'Onyx': 'Onyx', 'Bumblebee': 'Bumblebee', 'BlackBasta': 'BlackBasta', 
              'QUIETEXIT': 'UNC3524', 'Shadowpad, PlugX, Gunters': 'Moshen Dragon', 'PlugX': 'Mustang Panda', 'NetDooka': 'NetDooka', 
              'Jester Stealer': 'UAC-0104', 'Cuba': 'Cuba', 'Sysrv, XMRig': 'Sysrv', 'Sysrv': 'Sysrv', 'Facestealer': 'Facestealer', 
              'MyKLoadClient, Zupdax,  Downloader.Climax, RtlShare, PlugX, Deed RAT, BH_A006': 'Space Pirates', 'DeadBolt': 'DeadBolt', 
              'NukeSped, Jin Miner': 'Lazarus Group', 'XOR DDoS': 'XorDDoS AKA XOR DDoS', 'Cl0p': 'Cl0p', 
              'INDUSTROYER2, CADDYWIPER, ORCSHRED, SOLOSHRED, AWFULSHRED': 'Sandworm', 'Cheers': 'Cheers', 
              'BlackCat AKA ALPHV': 'BlackCat AKA ALPHV', 
              'Xloader': 'Roaming Mantis', 'CoolRAT': 'TA 413', 'GoodWill': 'GoodWill', 'SVCReady': 'TA551', 'Cuba, BUGHATCH': 'Cuba', 
              'Cobalt Strike, XMRig': 'Hezb', 'RansomHouse': 'RansomHouse', 'eCh0raix': 'eCh0raix', 'Hermit': 'Правительство Италии', 
              'Meteor': 'Gonjeshke Darande', 'Yanluowang': 'Yanluowang', 'More_Eggs': 'Golden Chickens', 'BlackCat': 'BlackCat AKA ALPHV', 
              'Scieron, HeaderTip': 'UAC-0026', 'Black Basta, Gh0st RAT': 'Black Basta', 'TRITON (TRISIS)': 'XENOTIME', 
              'Panchan, XMRig, nbhash': 'Panchan', 'SMSFactory': 'SMSFactory', 'CreepyDrive, CreepySnail': 'Polonium', 'Ousaban': 'Ousaban', 
              'Ragnar Locker': 'Ragnar Locker', 'CL0P': 'CL0P', 'Dracarys': 'Bitter', 'LazaSpy': 'APT36 AKA Transparent Tribe', 
              'Industrial Spy': 'Industrial Spy', 'LV': 'LV', 'FakeCrack': 'FakeCrack', 'AsyncRAT': 'Несколько субъектов угроз', 
              'WannaFriendMe': 'WannaFriendMe', 'CrescentImp': 'Sandworm', 'Cerber2021': 'Cerber2021', 'AvosLocker': 'AvosLocker', 
              'Hello XD': 'Hello XD', 'Night Sky': 'DEV-0234', 'XMRig, Cobalt Strike': 'Blue Mockingbird', 'MaliBot': 'MaliBot', 
              'Daixin Team': 'Daixin Team', 'SHARPEXT': 'Kimsuky', 'FakeUpdates, Raspberry Robin, Cobalt Strike': 'DEV-0206', 
              'BianLian': 'BianLian', 'Qakbot': 'Qakbot', 'Lilith': 'Lilith', 'WarzoneRAT': 'Confucius', 'Cobalt Strike': 'UAC-0098', 
              'Revive': 'Revive', 'Cobalt Strike, ShadowPad': 'Неназванная китайская АРТ-группировка', 'CredoMap': 'APT28 AKA Fancy Bear', 
              'CrimsonRAT, ObliqueRAT, CapraRAT': 'APT36 AKA Transparent Tribe', 'Chinoxy': 'ТА459', 'WatchDog': 'WatchDog', 
              'Nimbda, Yahoyah, Tclient': 'Tropic Trooper', 'Keona Clipper': 'Keona Clipper', 
              'RelicRace, RelicSource, Formbook, Snake Keylogger': 'UAC-0041', 'CosmicStrand': 'Китайскоязычные злоумышленники', 
              'Racoon Stealer': 'Raccoon Stealer', 'Karakurt': 'Karakurt', 'Ljl Backdoor': 'TAC-040', 
              'CharmPower, PINEFLOWER': 'APT35 AKA Charming Kitten', 'GwisinLocker': 'GwisinLocker', 'BlackByte': 'BlackByte', 
              'Joker': 'Joker', 'Facesteaer': 'Facesteaer', 'Coper': 'Coper', 'CloudMensis': 'CloudMensis', 
              'Grimplant, Graphsteel': 'Ghostwriter', 'Grimplant, Graphsteel, Remote Utilities': 'UNC2589', 'Luna': 'Luna', 
              'KonniRAT': 'APT37', 'GoMet': 'APT-группировка из России', 'Manjusaka, Cobalt Strike': 'Китайский злоумышленник', 
              'Amadey, SmokeLoader': 'Amadey', 'DevilsTongue': 'Candiru', 'EvilNum': 'TA4563', 'SolidBit': 'SolidBit', 'BRATA': 'BRATA', 
              'Bumblebee, Cobalt Strike, Diavol, Conti': 'Projector Libra AKA EXOTIC LILY', 'Orchard': 'Orchard', 
              'Donut Leaks': 'DonutLeaks', 
              'Cuba, ROMCOM RAT': 'Tropical Scorpius', 'RansomEXX': 'RansomEXX', 'SRG': 'Silent Ransom Group (SRG) AKA Luna Moth', 
              'SOVA': 'SOVA', 'BROWSER STEALER': 'APT-C-35', 'Everest': 'Everest', 
              'MidnightBlizzard': 'MidnightBlizzard (она же Nobelium, APT29 и Cozy Bear)',
              'Midnight Blizzard': 'Midnight Blizzard (она же Nobelium, APT29 и Cozy Bear)'}

def get_group(non_duplicates):
    
    group = []
    for key, value in group_dict.items():
        for malware in non_duplicates:
            if malware in key:
                group.append(f"{value}")
                
    drop_duplicates = []
    for vpo in list(set(group)):
        drop_duplicates.append(vpo)
        
    return drop_duplicates

def get_description(text):
    api_token = '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'
    url_summarization = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    payload = {"inputs": text}
    headers = {'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'}
    response = requests.post(url_summarization, json=payload, headers=headers)
    result = response.json()
    
    return result[0].get('summary_text')

total_dict = {'dcRAT': {'Name': 'dcRAT virus',
                        'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                        'Detection Names': 'AVG (Win32:Malware-gen), BitDefender (Trojan.GenericKD.32546165), ESET-NOD32 (A Variant Of  Generik.JTQNQBW), Kaspersky (Trojan.Win32.Vasal.adt), Full List (VirusTotal).',
                        'Malicious Process Name(s)': 'Randomly-named processes.',
                        'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                        'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                        'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
             'LockBit 3.0': {'Name': 'LockBit 3.0 virus',
                            'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                            'Encrypted Files Extension': 'Randomly generated extension',
                            'Ransom Demanding Message': '[random_string].README.txt',
                            'Free Decryptor Available?': 'Partial (more information below)',
                            'Cyber Criminal Contact': 'Chat on the provided websites',
                            'Detection Names': 'Avast (Win32:CrypterX-gen [Trj]), Combo Cleaner (Gen:Trojan.Heur.UT.kuW@aG4Vbyc), Emsisoft (Gen:Trojan.Heur.UT.kuW@aG4Vbyc (B)), Kaspersky (UDS:Trojan.Multi.GenericML.xnet), Microsoft (Trojan:Win32/Casdet!rfn), Full List Of Detections (VirusTotal)',
                            'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                            'Additional Information': 'Lockbit 3.0 is also known as LockBit Black. It is a new variant of theLockBit ransomware.',
                            'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                            'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
             'LockBit 2.0': {'Name': 'LockBit 2.0 virus',
                            'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                            'Encrypted Files Extension': '.lockbit',
                            'Ransom Demanding Message': 'Text presented in LockBit_Ransomware.hta, Restore-My-Files.txt, and desktop wallpaper',
                            'Cyber Criminal Contact': 'Websites on Tor network',
                            'Detection Names': 'Avast (Win32:LockBit-A [Ransom]), Combo Cleaner (Gen:Variant.Ransom.Lockbit2.9), ESET-NOD32 (A Variant Of Win32/Filecoder.Lockbit.E), Kaspersky (HEUR:Trojan-Ransom.Win32.Lockbit.gen), Microsoft (Ransom:Win32/Lockbit.STA), Full List Of Detections (VirusTotal)',
                            'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                            'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                            'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
             'AstraLocker': {'Name': 'AstraLocker virus',
                            'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                            'Encrypted Files Extension': 'Extension consisting of a random character string',
                            'Ransom Demanding Message': 'Recover_Files.txt',
                            'Ransom Amount': '50 USD in Bitcoin/Monero crypticurrency',
                            'Cyber Criminal Cryptowallet Addresses': '47moe29QP2xF2myDYaaMCJHpLGsXLPw17CqMQFeuB3NTzJ2X28tfRmWaPyPQgvoHVDUe4gP8h4w4pXCtX1gg7SpGAgh6qqS (Monero)17CqMQFeuB3NTzJ2X28tfRmWaPyPQgvoHV (Bitcoin)',
                            'Cyber Criminal Contact': 'AstraRansomware@protonmail.com',
                            'Detection Names': 'Avast (Win32:RansomX-gen [Ransom]), Combo Cleaner (Generic.Ransom.HydraCrypt.73F68097), ESET-NOD32 (A Variant Of MSIL/Filecoder.AGP), Kaspersky (HEUR:Trojan.MSIL.Fsysna.gen), Microsoft (Ransom:MSIL/ApisCryptor.PAA!MTB), Full List Of Detections (VirusTotal)',
                            'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                            'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                            'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
            'XMRig': {'Name': 'XMRig CPU Miner',
                      'Threat Type': 'Trojan, Crypto Miner',
                      'Detection Names': 'Avast (Win64:Trojan-gen), BitDefender (Trojan.GenericKD.43163708), ESET-NOD32 (Win64/CoinMiner.YZ), Kaspersky (HEUR:Trojan.Win32.Generic), Microsoft (Trojan:Win64/CoinMiner.WE), Full List Of Detections (VirusTotal)',
                      'Symptoms': 'Significantly decreased system performance, CPU resource usage.',
                      'Distribution methods': 'Deceptive pop-up ads, free software installers (bundling), fake flash player installers.',
                      'Damage': 'Decreased computer performance, browser tracking - privacy issues, possible additional malware infections.'},
            'LockBit': {'Name': 'LockBit virus',
                        'Threat Type': 'Ransomware, Crypto Virus, Files locker.',
                        'Encrypted Files Extension': '.abcd, lockbit',
                        'Ransom Demanding Message': 'Restore-My-Files.txt',
                        'Cyber Criminal Contact': 'goodmen@countermail.com and goodmen@cock.li email addresses, chat in Tor website',
                        'Detection Names': 'Avast (Win32:Malware-gen), BitDefender (Gen:Heur.Ransom.Imps.3), ESET-NOD32 (A Variant Of Win32/Filecoder.NXQ), Kaspersky (Trojan.Win32.DelShad.bqj), Full List Of Detections (VirusTotal)',
                        'Detection Names (updated variant)': 'Avast (Win32:Fraudo [Trj]), BitDefender (Gen:Heur.Ransom.Imps.1), ESET-NOD32 (A Variant Of Win32/Filecoder.NXQ), Kaspersky (Trojan.Win32.DelShad.chy), Full List Of Detections (VirusTotal)',
                        'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                        'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                        'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
         'Quantum': {'Name': 'Quantum virus',
                    'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                    'Encrypted Files Extension': '.quantum',
                    'Ransom Demanding Message': 'README_TO_DECRYPT.html',
                    'Free Decryptor Available?': 'No',
                    'Cyber Criminal Contact': 'Chat on the provided Tor website',
                    'Detection Names': 'Avast (Win32:RansomX-gen [Ransom]), Combo Cleaner (Gen:Trojan.Heur3.LPT.eCW@aSa3sMnab), ESET-NOD32 (A Variant Of Win32/Filecoder.MountLocker.E), Kaspersky (Trojan-Ransom.Win32.Encoder.ppn), Microsoft (Ransom:Win32/QuantumLocker.MAK!MTB), Full List Of Detections (VirusTotal)',
                    'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                    'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                    'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
                    'AvosLocker': {'Name': 'AvosLocker virus',
                    'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                    'Detection Names': 'Avast (Win32:Malware-gen), BitDefender (DeepScan:Generic.Ransom.BTCWare.AB3FFEB6), ESET-NOD32 (A Variant Of Win32/Filecoder.OHU), Kaspersky (HEUR:Trojan-Ransom.Win32.Cryptor.gen), Microsoft (Ransom:Win32/Avaddon.P!MSR), Full List Of Detections (VirusTotal)',
                    'Encrypted Files Extension': '.avos, .avos2',
                    'Ransom Demanding Message': 'GET_YOUR_FILES_BACK.txt',
                    'Ransom Amount': '888.89 XMR (Monero cryptocurrency)',
                    'Cyber Criminal Contact': 'Website on Tor network',
                    'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                    'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                    'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
        'Meterpreter': {'Name': 'Meterpreter malware',
                        'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                        'Detection Names (MS Office document)': 'Avast (VBA:Downloader-EON [Trj]), BitDefender (VB:Trojan.Valyria.447), ESET-NOD32 (VBA/TrojanDropper.Agent.UR), Kaspersky (HEUR:Trojan.Win32.Generic), Full List (VirusTotal)',
                        "Detection Names (Malicious Document's Payload)": 'Avast (Win32:SwPatch [Wrm]), BitDefender (Trojan.CryptZ.Gen), ESET-NOD32 (A Variant Of Win32/Rozena.ED), Kaspersky (HEUR:Trojan.Win32.Generic), Full List (VirusTotal)',
                        'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                        'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                        'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'Cobalt Strike': {'Name': 'Cobalt Strike virus',
                            'Threat Type': 'Trojan, Password stealing virus, Banking malware, Spyware',
                            'Detection Names (neskodnydrop.exe)': 'Avast (Win32:Malware-gen), BitDefender (Gen:Variant.Ursu.254544), ESET-NOD32 (A Variant Of Win32/RiskWare.CobaltStrike.Artifact.A), Kaspersky (HEUR:Trojan.Win32.Generic), Full List (VirusTotal)',
                            'Symptoms': "Trojans are designed to stealthily infiltrate victim's computer and remain silent thus no particular symptoms are clearly visible on an infected machine.",
                            'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, software cracks.',
                            'Damage': "Stolen banking information, passwords, identity theft, victim's computer added to a botnet."},
        'IcedID': {'Name': 'IcedID malware',
                    'Threat Type': 'Trojan, Password stealing virus, Banking malware, Spyware',
                    'Detection Names (Mittie resume.doc - malicious email attachment)': 'Avast (Other:Malware-gen [Trj]), BitDefender (Trojan.GenericKD.31904759), ESET-NOD32 (VBA/TrojanDownloader.Agent.NPX), Kaspersky (HEUR:Trojan-Downloader.MSOffice.SLoad.gen), Full List (VirusTotal)',
                    'Symptoms': "Trojans are designed to stealthily infiltrate victim's computer and remain silent thus no particular symptoms are clearly visible on an infected machine.",
                    'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, software cracks,Emotet trojan.',
                    'Damage': "Stolen banking information, passwords, identity theft, victim's computer added to a botnet."},
        'Forest': {'Name': 'Forest Wallpapers',
                    'Threat Type': 'Browser Hijacker, Redirect, Search Hijacker, Toolbar, Unwanted New Tab',
                    'Browser Extension(s)': 'Forest Wallpapers',
                    'Promoted URL': 'forestwallpapers.online',
                    'Detection Names (forestwallpapers.online)': 'N/A (VirusTotal)',
                    'Serving IP Address (forestwallpapers.online)': '85.159.209.37',
                    'Affected Browser Settings': 'Homepage, new tab URL, default search engine',
                    'Symptoms': "Manipulated Internet browser settings (homepage, default Internet search engine, new tab settings). Users are forced to visit the hijacker's website and search the Internet using their search engines.",
                    'Distribution methods': 'Deceptive pop-up ads, free software installers (bundling).',
                    'Damage': 'Internet browser tracking (potential privacy issues), display of unwanted ads, redirects to dubious websites.'},
        'Bumblebee': {'Name': 'Bumblebee loader',
                        'Threat Type': 'Malware loader',
                        'Detection Names (Malicious ISO File)': 'Avast (LNK:Agent-BD [Trj]), Combo Cleaner (Gen:Variant.Lazy.164691), ESET-NOD32 (A Variant Of Win64/Kryptik.CZJ), Kaspersky (HEUR:Trojan.Win32.Generic), Microsoft (Program:Win32/Wacapew.C!ml), Full List (VirusTotal)',
                        'Detection Names (Bumblebee)': 'Avast (Win32:TrojanX-gen [Trj]), Combo Cleaner (Trojan.GenericKD.50206978), ESET-NOD32 (A Variant Of Win64/Agent.BEG), Kaspersky (UDS:Trojan.Win64.Shelma.a), Microsoft (Trojan:Win32/Casdet!rfn), Full List (VirusTotal)',
                        'Payload': 'Cobalt Strike,ransomware, and possibly other malware',
                        'Symptoms': "Bumblebee is designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine. It also can avoid detecion and analysis.",
                        'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                        'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
        'RedAlert': {'Name': '"RedAlert - Rocket Alerts App" malware',
                    'Threat Type': 'Android malware, spyware, malicious application.',
                    'Detection Names': 'BitDefenderFalx (Android.Riskware.Agent.gHWCK), DrWeb (Android.Spy.1169.origin), Kaspersky (HEUR:Trojan.AndroidOS.Piom.azwx), McAfee (Artemis!410C6E3AF93A), Symantec Mobile Insight (AdLibrary:Generisk), ZoneAlarm by Check Point (HEUR:Trojan.AndroidOS.Piom.azwx), Full List (VirusTotal)',
                    'Symptoms': "The device is running slow, system settings are modified without user's permission, questionable applications appear, data and battery usage is increased significantly.",
                    'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, deceptive applications, scam websites.',
                    'Damage': 'Stolen personal information (private messages, logins/passwords, etc.), decreased device performance, battery is drained quickly, decreased Internet speed, huge data losses, monetary losses, stolen identity (malicious apps might abuse communication apps).'},
        'ZxxZ': {'Name': 'ZxxZ malware',
                  'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                  'Detection Names (ZxxZ)': 'Avast (Win32:Trojan-gen), Combo Cleaner (Gen:Variant.Doina.23073), ESET-NOD32 (A Variant Of Win32/TrojanDownloader.Small.BJH), Kaspersky (HEUR:Trojan.Win32.APosT.gen), Microsoft (TrojanDownloader:Win32/FraudLoad.AN!), Full List Of Detections (VirusTotal)',
                  'Detection Names (malicious doc)': 'Avast (Other:Malware-gen [Trj]), Combo Cleaner (Trojan.GenericKD.47845687), ESET-NOD32 (Win32/Exploit.CVE-2017-11882.CO), Kaspersky (HEUR:Exploit.MSOffice.CVE-2018-0802.gen), Microsoft (Exploit:O97M/CVE-2017-11882.SM!MTB), Full List Of Detections (VirusTotal)',
                  'Detection Names (malicious excel)': 'Avast (Other:Malware-gen [Trj]), Combo Cleaner (Trojan.Generic.31270685), ESET-NOD32 (Win32/Exploit.CVE-2018-0798.A), Kaspersky (HEUR:Exploit.MSOffice.CVE-2018-0802.gen), Microsoft (Exploit:O97M/CVE-2018-0802.AL!MTB), Full List Of Detections (VirusTotal)',
                  'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                  'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                    'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'Brute Ratel': {'Name': 'Brute Ratel post-exploitation toolkit',
                        'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                        'Detection Names (Roshan_CV.iso)': 'Avast (Win64:Malware-gen), Combo Cleaner (Trojan.Brutel.A), ESET-NOD32 (A Variant Of Win64/Agent.BMC), Kaspersky (Trojan.Win32.Agent.xapvcq), Microsoft (Trojan:Win32/BruteRatel!MSR), Full List (VirusTotal)',
                        'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                        'Distribution methods': 'ISO file containing a malicious LNK (shortcut) file with a fake MS Office Word icon.',
                        'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet, data encryption, monetary loss."},
         'Luca Stealer': {'Name': 'Luca (RSStealer) malware',
                          'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                            'Detection Names': 'Avast (Win64:PWSX-gen [Trj]), Combo Cleaner (Gen:Variant.Tedy.168929), Fortinet (W32/PossibleThreat), Kaspersky (Trojan-PSW.MSIL.Reline.orm), Microsoft (Trojan:Win32/Wacatac.B!ml), Full List Of Detections (VirusTotal)',
                          'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                          'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                          'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'Predator': {'Name': 'Predator The Thief trojan',
                      'Threat Type': 'Trojan, Password stealing virus, Banking malware, Spyware.',
                      'Detection Names': 'Avast (Win32:DangerousSig [Trj]), BitDefender (Trojan.GenericKD.31830202), ESET-NOD32 (Win32/Spy.Agent.PQW), Kaspersky (Trojan-Spy.Win32.Stealer.lbn), Full List (VirusTotal)',
                      'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent. Thus, no particular symptoms are clearly visible on an infected machine.",
                      'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, software cracks.',
                      'Damage': "Stolen banking information, passwords, identity theft, victim's computer added to a botnet."},
         'GootKit': {'Name': 'GootKit virus',
                      'Threat Type': 'Trojan, Password stealing virus, Banking malware, Spyware',
                      'Name Of Malicious Attachment': 'gootkit-samples.zip',
                      'Detection Names': 'Avast (VBS:Agent-BUK [Trj]), ESET-NOD32 (VBA/TrojanDownloader.Agent.NJN), Fortinet (WM/Agent.7319!tr), Kaspersky (HEUR:Trojan.MSOffice.SAgent.gen), Full List (VirusTotal)',
                      'Malicious Process Name': 'Standinstrument (the name may vary)',
                      'Symptoms': "Trojans are designed to stealthily infiltrate victim's computer and remain silent thus no particular symptoms are clearly visible on an infected machine.",
                      'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, software cracks.',
                      'Damage': "Stolen banking information, passwords, identity theft, victim's computer added to a botnet."},
         'Colibri': {'Name': 'Colibri malware loader',
                      'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                      'Detection Names': 'Avast (Win32:Colibri-A [Drp]), Combo Cleaner (Gen:Variant.Razy.942612), ESET-NOD32 (A Variant Of Win32/Agent.ADQQ), Kaspersky (Trojan-Spy.Win32.SpyEyes.bslf), Microsoft (Trojan:Win32/Tiggre!rfn), Full List (VirusTotal)',
                      'Symptoms': "Loaders are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                      'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                      'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'Vidar': {'Name': 'Vidar Stealer',
                    'Threat Type': 'Trojan, Password stealing virus, Banking malware, Spyware',
                    'Malicious Filename': 'verinstere.xls',
                    'Websites Spreading Vidar By Disguising It As Legitimate Software': 'dumepad[.]com, hurimis[.]com, kisomer[.]com, metersik[.]com, nuktose[.]com, nviktus[.]com, opriky[.]com, vikolin[.]com, viulinik[.]com, kulinkos[.]com, crypto-widget[.]live, download-best[.]com, github[.]llc, intuitquickbooks[.]space',
                    'Detection Names': 'Avast (Win32:Malware-gen), BitDefender (Trojan.GenericKD.32180278), ESET-NOD32 (A Variant Of Win32/Kryptik.GUVT), Kaspersky (Trojan.Win32.Chapak.dwtd), Full List (VirusTotal)',
                    'Malicious Process Name(s)': 'Deligthers Simulations Retriever... (the process name may vary).',
                    'Symptoms': "Trojans are designed to stealthily infiltrate victim's computer and remain silent thus no particular symptoms are clearly visible on an infected machine.",
                    'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, software cracks.',
                    'Damage': "Stolen banking information, passwords, identity theft, victim's computer added to a botnet."},
         'FFDroid': {'Name': 'FFDroider malware',
                      'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                      'Detection Names': 'Avast (Win32:Trojan-gen), Combo Cleaner (Trojan.GenericKD.48533931), ESET-NOD32 (A Variant Of Win32/PSW.Agent.OHG), Kaspersky (Trojan-Banker.Win32.Passteal.sn), Microsoft (Trojan:Win32/Passteal.MA!MTB), Full List Of Detections (VirusTotal)',
                      'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                      'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                      'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'NetSupport RAT': {'Name': 'NetSupport Manager remote access tool',
                            'Threat Type': 'Remote access tool, trojan, spyware.',
                            'Detection Names (client32.exe)': 'DrWeb (Program.RemoteAdmin.837), Fortinet (Riskware/RemoteAdmin), Kaspersky (not-a-virus:RemoteAdmin.Win32.NetSup.i), Full List (VirusTotal)',
                            'Related Domain(s)': 'bl0kchain[.]review, bl0kchain[.]stream, bl0kchain[.]win, desjardinscourriel818654[.]pw, desjardinscourriel8as4363[.]pw, desjardinscourrielf36ws[.]pw, desjardinsmail6as6545g[.]pw, desjardinsmail6sa4524[.]pw lalka14881112[.]asyx[.]ru, otedehea[.]review',
                            'Symptoms': "Remote access tools allow criminals to remotely manipulate the system and perform various tasks without users' consent.",
                            'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, software cracks, bundling.',
                            'Damage': "Stolen banking information, passwords, identity theft, victim's computer added to a botnet, malware infections."},
         'SharkBot': {'Name': 'SharkBot virus',
                      'Threat Type': 'Android malware, malicious application, unwanted application.',
                      'Detection Names': 'Avast (Android:SharkBot-A [Bank]), Combo Cleaner (Trojan.GenericKD.47387732), ESET-NOD32 (Android/Spy.Agent.BWR), Kaspersky (HEUR:Trojan-Banker.AndroidOS.Sharkbot.a), Full List (VirusTotal)',
                      'Symptoms': "The device is running slow, system settings are modified without user's permission, questionable applications appear, data and battery usage is increased significantly, browsers redirect to questionable websites, intrusive advertisements are delivered.",
                      'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, deceptive applications, scam websites.',
                      'Damage': 'Stolen personal information (private messages, logins/passwords, etc.), decreased device performance, battery is drained quickly, decreased Internet speed, huge data losses, monetary losses, stolen identity (malicious apps might abuse communication apps).'},
         'BlackCat': {'Name': 'ALPHV (BlackCat) virus',
                      'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                      'Encrypted Files Extension': 'Depends on the variant',
                      'Ransom Demanding Message': 'GET IT BACK-[file_extension]-FILES.txt',
                      'Free Decryptor Available?': 'No',
                      'Cyber Criminal Contact': 'Website on Tor network',
                      'Detection Names (Windows)': 'Avast (Win32:Malware-gen), Combo Cleaner (Trojan.GenericKD.38153014), Kaspersky (UDS:Trojan.Win32.Agentb.a), Malwarebytes (Malware.AI.2115381737), Microsoft (Trojan:Win32/Woreflint.A!cl), Full List Of Detections (VirusTotal)',
                      'Detection Names (Linux)': 'McAfee-GW-Edition (Artemis), Microsoft (Ransom:Linux/BlackCat.A!MTB), TrendMicro (Ransom.Linux.BLACKCAT.YXCDFZ), Full List Of Detections (VirusTotal)',
                      'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                      'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                      'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
         'NB65': {'Name': 'NB65 virus',
                  'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                  'Encrypted Files Extension': '.NB65',
                  'Ransom Demanding Message': 'R3ADM3.txt',
                  'Free Decryptor Available?': 'No',
                  'Cyber Criminal Contact': 'network_battalion_0065@riseup.net',
                  'Detection Names': 'Avast (Win32:Conti-B [Ransom]), Combo Cleaner (Gen:Variant.Cerbu.84170), ESET-NOD32 (A Variant Of Win32/Filecoder.Conti.K), Kaspersky (HEUR:Trojan-Ransom.Win32.Conti.gen), Microsoft (Ransom:Win32/Conti.AD!MTB), Full List Of Detections (VirusTotal)',
                  'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                  'Additional Information': 'NB65 is based onCONTIransomware',
                  'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                  'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
         'META Stealer': {'Name': 'MetaStealer information stealer',
                          'Threat Type': 'Information stealer',
                          'Detection Names': 'Avast (MacOS:Agent-XZ [Trj]), Combo Cleaner (Trojan.Generic.33936513), Emsisoft (Trojan.Generic.33936513 (B)), Kaspersky (UDS:Trojan-PSW.OSX.HashBreaker.c), Symantec (OSX.Trojan.Gen), Full List (VirusTotal)',
                          'Symptoms': "Information stealers are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                          'Distribution methods': "Fake applications, DMG files obtained from shady sources, infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                          'Damage': 'Stolen passwords and banking information, identity theft, financial loss, and more.',
                          'Malware Removal (Mac)': 'To eliminate possible malware infections, scan your Mac with legitimate antivirus software. Our security researchers recommend using Combo Cleaner.▼ Download Combo Cleaner for MacTo use full-featured product, you have to purchase a license for Combo Cleaner. Limited seven days free trial available. Combo Cleaner is owned and operated by Rcs Lt, the parent company of PCRisk.comread more.'},
         'Fakecalls': {'Name': 'Fakecalls Android malware',
                       'Threat Type': 'Android malware, malicious application, unwanted application.',
                       'Detection Names': 'Avast (Android:Fakecalls-H [Trj]), BitDefenderFalx (Android.Trojan.Banker.XK), ESET-NOD32 (A Variant Of Android/Spy.Banker.BAD), Kaspersky (HEUR:Trojan-Banker.AndroidOS.Fakecalls.h), Full List (VirusTotal)',
                        'Symptoms': "The device is running slow, system settings are modified without user's permission, questionable applications appear, data and battery usage is increased significantly",
                        'Distribution methods': 'Fake banking applications (e.g., Kookbik Bank, KakaoBank)',
                        'Damage': 'Stolen personal information (private messages, logins/passwords, etc.), decreased device performance, battery is drained quickly, decreased Internet speed, monetary losses, stolen identity.'},
         'Black Basta, Black': {'Name': 'Black Basta virus',
                                'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                                'Encrypted Files Extension': '.basta',
                                'Ransom Demanding Message': 'readme.txt',
                                'Ransom Amount': '2.7 million USD (may vary)',
                                'Free Decryptor Available?': 'No',
                                'Cyber Criminal Contact': 'Chat on Tor network website',
                                'Detection Names': 'Avast (Win32:Malware-gen), Combo Cleaner (Gen:Heur.Ransom.REntS.Gen.1), ESET-NOD32 (Win32/Filecoder.OKW), Kaspersky (HEUR:Trojan.Win32.DelShad.gen), Microsoft (Trojan:Win32/Sabsik.FL.B!ml), Full List Of Detections (VirusTotal)',
                                'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                                'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                                'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
         'CaddyWiper': {'Name': 'CaddyWiper virus',
                        'Threat Type': 'Trojan, Data Wiper',
                        'Detection Names': 'Avast (Win32:Malware-gen), Combo Cleaner (Gen:Variant.Razy.728059), ESET-NOD32 (Win32/KillDisk.NCX), Kaspersky (HEUR:Trojan.Win32.Generic), Microsoft (DoS:Win32/CaddyBlade.A!dha), Full List Of Detections (VirusTotal)',
                        'Symptoms': 'System cannot be booted, operating system is undetected.',
                        'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                        'Damage': 'Permanent data loss, inoperable device.'},
        'ZingoStealer': {'Name': 'Ginzo (ZingoStealer) information stealer',
                        'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                        'Detection Names': 'Avast (Win32:TrojanX-gen [Trj]), Combo Cleaner (Gen:Variant.Cerbu.134541), ESET-NOD32 (A Variant Of MSIL/Spy.Agent.DUZ), Kaspersky (HEUR:Trojan-PSW.MSIL.Stealer.gen), Microsoft (Trojan:Win32/Sabsik.FL.B!ml), Full List (VirusTotal)',
                        'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                        'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                        'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'TrickBot': {'Name': 'TrickBot malware',
                      'Threat Type': 'Trojan, Password stealing virus, Banking malware, Spyware',
                      'Detection Names': 'Avast (Win32:Malware-gen), BitDefender (Trojan.Agent.CWSV), ESET-NOD32 (Win32/TrickBot.AJ), Kaspersky (Trojan.Win32.Mansabo.awr), Full List (VirusTotal)',
                        'Symptoms': "Trojans are designed to stealthily infiltrate victim's computer and remain silent thus no particular symptoms are clearly visible on an infected machine.",
                        'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, software cracks.',
                        'Damage': "Stolen banking information, passwords, identity theft, victim's computer added to a botnet."},
         'MSPLT virus': {'Name': 'MSPLT virus',
                         'Threat Type': 'Ransomware, Crypto Virus, Files locker.',
                         'Encrypted Files Extension': ".MSPLT (files are also appended with a unique ID and cyber criminals' email address).",
                          'Ransom Demand Message': 'Text presented in the pop-up and FILES ENCRYPTED.txt file.',
                          'Cyber Criminal Contact': 'supermetasploit@aol.com and supermetasploit@cock.li',
                          'Detection Names': 'Avast (Win32:RansomX-gen [Ransom]), BitDefender (Trojan.Ransom.Crysis.E), ESET-NOD32 (A Variant Of Win32/Filecoder.Crysis.P), Kaspersky (Trojan-Ransom.Win32.Crusis.to), Full List Of Detections (VirusTotal)',
                          'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                            'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                            'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing Trojans and malware infections can be installed together with a ransomware infection.'},
         'Onyx': {'Name': 'ONYX virus',
                  'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                  'Encrypted Files Extension': 'Randomly generated extension',
                  'Ransom Demanding Message': 'readme.txt',
                  'Free Decryptor Available?': 'No',
                  'Cyber Criminal Contact': 'Chat on the provided Tor website',
                  'Detection Names': 'Avast (Win32:RansomX-gen [Ransom]), Combo Cleaner (IL:Trojan.MSILZilla.5554), ESET-NOD32 (A Variant Of MSIL/Filecoder.AGP), Kaspersky (HEUR:Trojan-Ransom.MSIL.Agent.gen), Microsoft (Ransom:MSIL/FileCoder.AD!MTB), Full List Of Detections (VirusTotal)',
                  'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                  'Additional Information': 'ONYX deletes files larger than 200MB and replaces them with random files. Also, it downloads all data before encrypting it.',
                  'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                  'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
         'PlugX': {'Name': 'PlugX remote access trojan',
                    'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                    'Detection Names': 'Avast (FileRepMalware), BitDefender (Trojan.GenericKD.3613716), ESET-NOD32 (A Variant Of Win32/Korplug.CV), Kaspersky (Backdoor.Win32.Gulpix.xst), Full List (VirusTotal)',
                    'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                    'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                    'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'NetDooka': {'Name': 'NetDooka remote access trojan',
                      'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                      'Detection Names (NetDooka)': 'Avast (Win32:PWSX-gen [Trj]), Combo Cleaner (IL:Trojan.MSILZilla.10987), ESET-NOD32 (MSIL/Agent.DQV), Kaspersky (HEUR:Trojan-PSW.MSIL.Stealer.gen), Microsoft (Trojan:Win32/Woreflint.A!cl), Full List Of Detections (VirusTotal)',
                      'Detection Names (PrivateLoader)': 'Avast (Win32:PWSX-gen [Trj]), Combo Cleaner (Trojan.Ransom.GenericKD.39505729), ESET-NOD32 (A Variant Of Win32/Kryptik.HPFX), Kaspersky (HEUR:Trojan.Win32.Zapchast.gen), Microsoft (Ransom:Win32/StopCrypt.PBH!MTB), Full List Of Detections (VirusTotal)',
                        'Related Domains': 'data-file-data-18[.]com, file-coin-coin-10[.]com',
                        'Detection Names (data-file-data-18[.]com)': 'Combo Cleaner (Malware), ESET (Malware), Fortinet (Malware), Heimdal Security (Malicious), Sophos (Malware), Full List Of Detections (VirusTotal)',
                        'Detection Names (file-coin-coin-10[.]com)': 'Combo Cleaner (Malware), Dr.Web (Malicious), ESET (Malware), Fortinet (Malware), Heimdal Security (Malicious), Full List Of Detections (VirusTotal)',
                        'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                        'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                        'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'Jester Stealer': {'Name': 'Jester Stealer virus',
                            'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                            'Detection Names': 'Avast (Win32:PWSX-gen [Trj]), Combo Cleaner (Trojan.GenericKD.38009735), ESET-NOD32 (A Variant Of MSIL/Spy.Agent.AES), Kaspersky (HEUR:Trojan-PSW.MSIL.Reline.gen), Microsoft (Trojan:Win32/Stealer!MSR), Full List Of Detections (VirusTotal)',
                            'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                            'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                            'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'Joker': {'Name': 'Joker virus',
                    'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                    'Encrypted Files Extension': ".Joker (files are also appended with a unique ID and the cyber criminals' email)",
                    'Ransom Demanding Message': 'Decryption-Guide.HTA and Decryption-Guide.txt',
                    'Free Decryptor Available?': 'No',
                    'Cyber Criminal Contact': 'suppransomeware@tutanota.com, suppransomeware@mailfence.com',
                    'Detection Names': 'Avast (Win32:RansomX-gen [Ransom]), Combo Cleaner (DeepScan:Generic.Ransom.AmnesiaE.C4B18), ESET-NOD32 (A Variant Of Win32/Filecoder.Ouroboros.G), Kaspersky (HEUR:Trojan-Ransom.Win32.Generic), Microsoft (Trojan:Win32/Wacatac.B!ml), Full List Of Detections (VirusTotal)',
                    'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                    'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                    'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
         'Nerbian RAT': {'Name': 'Nerbian remote administration trojan',
                        'Threat Type': 'Remote Access Trojan',
                        'Detection Names (Malicious Document)': 'Avast (Other:Malware-gen [Trj]), Combo Cleaner (Trojan.Groooboor.Gen.12), ESET-NOD32 (DOC/TrojanDownloader.Agent.CF), Kaspersky (HEUR:Trojan-Downloader.MSOffice.Dotmer.gen), Tencent (Trojan.Win32.Office_Dl.11020340), Full List (VirusTotal)',
                        'Detection Names (Malware Dropper)': 'Avast (Win64:Malware-gen), Combo Cleaner (Trojan.GenericKD.50215192), ESET-NOD32 (WinGo/Agent.GF), Kaspersky (Trojan.Win32.Khalesi.lxrs), Microsoft (Trojan:Win32/Tnega!MSR), Full List (VirusTotal)',
                        'Detection Names (Nerbian RAT)': 'Avast (FileRepMalware [Misc]), Combo Cleaner (Trojan.GenericKD.50213936), ESET-NOD32 (A Variant Of WinGo/Packed.Obfuscated.A Suspicious), Kaspersky (UDS:Trojan.Multi.GenericML.xnet), Microsoft (Trojan:Win32/Tnega!MSR), Full List (VirusTotal)',
                        'Symptoms': "Most RATs are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                        'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                        'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'RedLine': {'Name': 'RedLine Stealer virus',
                    'Threat Type': 'Password-stealing virus, banking malware, spyware.',
                    'Detection Names': 'Avast (Win32:DropperX-gen [Drp]), BitDefender (Trojan.GenericKD.33518015), ESET-NOD32 (A Variant Of MSIL/TrojanDownloader.Agent.GAO), Kaspersky (HEUR:Trojan-Downloader.MSIL.Seraph.gen), Full List (VirusTotal)',
                    'Malicious Process Name(s)': 'AddInProcess.exe',
                    'Payload': 'RedLine Stealer can be used to spread a variety of malicious programs.',
                    'Symptoms': "Software of this kind is designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                    'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                    'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'BitRAT': {'Name': 'BitRAT remote access trojan',
                    'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                    'Detection Names': 'Avast (Win32:MalwareX-gen [Trj]), BitDefender (Gen:Heur.Conjar.6), ESET-NOD32 (A Variant Of Win32/Agent.ACBZ), Kaspersky (HEUR:Backdoor.Win32.Agent.gen), Full List (VirusTotal).',
                    'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                    'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                    'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'FaceStealer': {'Name': 'FaceStealer malware',
                        'Threat Type': 'Android malware, malicious application, unwanted application.',
                        'Detection Names': 'Avast-Mobile (Android:Evo-gen [Trj]), BitDefenderFalx (Android.Adware.Agent.BL), ESET-NOD32 (Multiple Detections), Kaspersky ( HEUR:Trojan-PSW.AndroidOS.Facestealer.x), Full List (VirusTotal)',
                        'Symptoms': "The device is running slow, system settings are modified without user's permission, questionable applications appear, data and battery usage is increased significantly, browsers redirect to questionable websites, intrusive advertisements are delivered.",
                        'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, deceptive applications, scam websites.',
                        'Damage': 'Stolen personal information (private messages, logins/passwords, etc.), decreased device performance, battery is drained quickly, decreased Internet speed, huge data losses, monetary losses, stolen identity (malicious apps might abuse communication apps).'},
         'NukeSped': {'Name': 'NUKESPED backdoor Trojan',
                        'Threat Type': 'Trojan',
                        'Detection Names': 'Avast (MacOS:NukeSpeed-C [Trj]), BitDefender (Trojan.MAC.Lazarus.C), ESET-NOD32 (OSX/NukeSped.C), Kaspersky (HEUR:Trojan-Dropper.OSX.Agent.d), Full List (VirusTotal)',
                        'Malicious Process Name': '.Flash Player',
                        'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                        'Distribution methods': "Infected email attachments, fake Flash Player installers or updaters, malicious online advertisements, social engineering, software 'cracks'.",
                        'Damage': "Stolen passwords and banking information, identity theft, victim's computer added to a botnet, installation of other malware."},
         'XLoader': {'Name': 'XLoader virus',
                    'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                    'Detection Names': 'Avast (Win32:MalwareX-gen [Trj]), BitDefender (Trojan.GenericKD.46631605), ESET-NOD32 (A Variant Of MSIL/GenKryptik.FHOR), Kaspersky (HEUR:Trojan-PSW.MSIL.Agensla.gen), Microsoft (Trojan:Win32/AgentTesla!ml), Full List Of Detections (VirusTotal)',
                    'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                    'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                    'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'GuLoader': {'Name': 'GuLoader (CloudEyE) downloader',
                        'Threat Type': 'Malware downloader',
                        'Detection Names': 'Avast (Win32:Trojan-gen), BitDefender (Trojan.GenericKD.33531900), ESET-NOD32 (A Variant Of Win32/Injector.EKYU), Kaspersky (Backdoor.MSIL.NanoBot.bbks), Full List (VirusTotal)',
                        'Payload': 'GuLoader can be used to infect systems withAgent Tesla,FormBook,LokiBot,NetWire,Remcos,Vidarand other malicious programs',
                        'Symptoms': "Programs like GuLoader are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                        'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                        'Damage': 'Stolen passwords and banking information, identity theft, data and/or monetary loss, problems with online privacy'},
         'YTStealer': {'Name': 'YTStealer virus',
                        'Threat Type': 'Trojan, password-stealing virus.',
                        'Detection Names': 'Avast (FileRepMalware [Misc]), Combo Cleaner (Trojan.GenericKDZ.87756), ESET-NOD32 (A Variant Of WinGo/Agent.FP), Kaspersky (Trojan.Win32.Khalesi.lypf), Microsoft (Trojan:Win32/Trickbot!ml), Full List Of Detections (VirusTotal)',
                        'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                        'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                        'Damage': 'Stolen YouTube accounts and related information, financial losses, identity theft.'},
         'Yanluowang': {'Name': 'Yanluowang virus',
                        'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                        'Encrypted Files Extension': '.yanluowang',
                        'Ransom Demanding Message': 'README.txt',
                        'Free Decryptor Available?': 'Yes, it can be downloadedhere',
                        'Cyber Criminal Contact': 'cang.leen@mailfence.com, yan.laowang@mailfence.com',
                        'Detection Names': 'Avast (Win32:Malware-gen), Combo Cleaner (Trojan.GenericKD.38174063), ESET-NOD32 (A Variant Of Win32/Filecoder.OJO), Kaspersky (HEUR:Trojan-Ransom.Win32.Agent.gen), Microsoft (Ransom:Win32/Yanluow.A), Full List Of Detections (VirusTotal)',
                        'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files',
                        'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads',
                        'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection'},
         'More_Eggs': {'Name': 'More_eggs trojan',
                        'Threat Type': 'Trojan, Password stealing virus, Banking malware, Spyware',
                        'Symptoms': "Trojans are designed to stealthily infiltrate victim's computer and remain silent thus no particular symptoms are clearly visible on an infected machine.",
                        'Distribution methods': 'Infected email attachments (e.g., malicious PDF and Word documents), malicious online advertisements, social engineering, software cracks.',
                        'Damage': "Stolen banking information, passwords, identity theft, victim's computer added to a botnet, additional infections, data loss, privacy issues, and more."},
         'CopperStealer': {'Name': 'CopperStealer virus',
                            'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                            'Detection Names': 'Avast (Win32:Trojan-gen), BitDefender (Gen:Variant.Zusy.356181), ESET-NOD32 (A Variant Of Win32/TrojanDropper.Agent.SML), Kaspersky (HEUR:Trojan-Dropper.Win32.Dapato.pef), Microsoft (Trojan:Win32/Tnega!ml), Full List Of Detections (VirusTotal)',
                            'Malicious Process Name(s)': 'DeltaCopy Server Console (process name may vary)',
                            'Related Domains': 'keyninja[.]com, startcrack[.]com, piratewares[.]com',
                            'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                            'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                            'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'Matanbuchus': {'Name': 'Matanbuchus virus',
                          'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                            'Detection Names': 'Avast (Win32:Malware-gen), Combo Cleaner (Gen:Variant.Graftor.928390), ESET-NOD32 (A Variant Of Win32/Kryptik.HJPA), Kaspersky (Trojan.Win32.Agentb.klom), Microsoft (Trojan:MSIL/Cryptor), Full List Of Detections (VirusTotal)',
                            'Detection Names (malicious Excel file)': 'Avast (VBS:Malware-gen), Combo Cleaner (Trojan.DOC.Agent.AXW), ESET-NOD32 (DOC/TrojanDownloader.Agent.DPA), Microsoft (TrojanDownloader:O97M/EncDoc.SME!MT), Full List Of Detections (VirusTotal)',
                            'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                            'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                            'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'Ousaban': {'Name': 'Javali (Ousaban) banking malware',
                    'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                    'Detection Names': 'Avast (Win32:Trojan-gen), BitDefender (Gen:Variant.Zusy.357421), ESET-NOD32 (A Variant Of Win32/Spy.Ousaban.B), Kaspersky (HEUR:Trojan-Banker.Win32.Javali.gen), Symantec (Trojan.Gen.MBT), Full List (VirusTotal)',
                    'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                    'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                      'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'X-FILES, XFiles': {'Name': 'X-FILES (XFiles) information stealer',
                             'Threat Type': 'Password-stealing virus, banking malware, information stealer.',
                             'Detection Names': 'Avast (Win32:MalwareX-gen [Trj]), BitDefender (Gen:Variant.Bulz.398877), ESET-NOD32 (A Variant Of MSIL/PSW.Agent.RXF), Kaspersky (UDS:Trojan-PSW.MSIL.Agent.a), Microsoft (PWS:MSIL/Browsstl.GA!MTB), Full List (VirusTotal)',
                              'Malicious Process Name(s)': 'Svc_host (its name may vary)',
                              'Symptoms': "Data stealers are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                              'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                              'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'Dracarys': {'Name': 'Dracarys spyware',
                      'Threat Type': 'Android malware, spyware',
                      'Related Domain': 'signalpremium[.]com',
                      'Detection Names (signalpremium[.]com)': 'Avira (Malware), Combo Cleaner (Malware), ESTsecurity (Malicious), Sophos (Malware), Full List (VirusTotal)',
                      'Detection Names (Dracarys)': 'Avast-Mobile (APK:RepMalware [Trj]), DrWeb (Android.Spy.1037.origin), ESET-NOD32 (A Variant Of Android/Spy.Dracarys.A), Kaspersky (HEUR:Trojan-Spy.AndroidOS.Dracarys.a), Full List (VirusTotal)',
                      'Symptoms': "The device is running slow, system settings are modified without user's permission, questionable applications appear, data and battery usage is increased significantly, browsers redirect to questionable websites, files are deleted.",
                      'Distribution methods': 'Fake applications hosted by deceptive websites.',
                      'Damage': 'Stolen personal information (private messages, call logs, etc.), decreased device performance, battery is drained quickly, decreased Internet speed, huge data losses, monetary losses, stolen identity.'},
         'AsyncRat, AsyncRat': {'Name': 'Async remote access trojan',
                                  'Threat Type': 'Remote Access Trojan',
                                  'Detection Names': 'AegisLab (Trojan.Win32.Generic.4!c), BitDefender (Gen:Variant.MSILPerseus.191214), ESET-NOD32 (A Variant Of MSIL/Agent.BVF), Kaspersky (HEUR:Trojan.Win32.Generic), Full List (VirusTotal)',
                                  'Symptoms': "Remote access trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                                  'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                                  'Damage': 'Stolen banking information, passwords, identity theft, installation of other malicious software'},
         'WANNAFRIENDME, WannaFriendMe': {'Name': 'WANNAFRIENDME 2 virus',
                                          'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                                          'Encrypted Files Extension': '.iRazormind',
                                          'Ransom Demanding Message': 'README.txt',
                                          'Free Decryptor Available?': 'No',
                                          'Ransom Amount': '800 Roblox gamepasses purchased with Robux',
                                          'Cyber Criminal Contact': 'sethprobest9008@gmail.com',
                                          'Detection Names': 'Avast (Win32:RansomX-gen [Ransom]), Combo Cleaner (IL:Trojan.MSILZilla.16368), ESET-NOD32 (A Variant Of MSIL/Filecoder.AK), Kaspersky (HEUR:Trojan-Ransom.MSIL.Encoder.gen), Microsoft (Ransom:MSIL/Ryzerlo.A), Full List Of Detections (VirusTotal)',
                                          'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                                          'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                                          'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
         'MaliBot': {'Name': 'MaliBot malware',
                      'Threat Type': 'Android malware, malicious application, unwanted application.',
                      'Detection Names': 'Avast-Mobile (APK:RepMalware [Trj]), ESET-NOD32 (Android/Spy.Agent.CAJ), Fortinet (Android/Agent.CAJ!tr.spy), Kaspersky (HEUR:Trojan-Banker.AndroidOS.Sova.b), Full List (VirusTotal)',
                      'Symptoms': "The device is running slow, system settings are modified without user's permission, questionable applications appear, data and battery usage is increased significantly.",
                      'Distribution methods': 'Malicious text messages (SMSes), infected email attachments, malicious online advertisements, social engineering, deceptive applications, scam websites.',
                      'Damage': 'Stolen personal information (private messages, logins/passwords, etc.), decreased device performance, battery is drained quickly, decreased Internet speed, huge data losses, monetary losses, stolen identity (malicious apps might abuse communication apps).'},
         'Raspberry Robin': {'Name': 'Raspberry Robin worm',
                              'Threat Type': 'Worm',
                              'Detection Names': 'Avast (Win32:Evo-gen [Trj]), Combo Cleaner (Gen:Variant.Lazy.261331), ESET-NOD32 (A Variant Of Win32/Injector.ERZV), Kaspersky (UDS:DangerousObject.Multi.Generic), Microsoft (Backdoor:Win32/RaspberryRobin.PA!MTB), Full List (VirusTotal)',
                              'Payload': 'Various malware, including ransomware and Trojans such as Clop and IceID',
                              'Symptoms': "Raspberry Robin is designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                              'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                              'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'HiddenAds, Hidden Ads': {'Name': 'HiddenAds trojan',
                                    'Threat Type': 'Android malware, malicious application, adware.',
                                    'Detection Names': 'Avast-Mobile (Android:Evo-gen [Trj]), ESET-NOD32 (A Variant Of Android/Hiddad.AQS), Kaspersky (Not-a-virus:HEUR:AdWare.AndroidOS.Teddad.i), McAfee (Artemis!C354D5529BEA), Full List (VirusTotal)',
                                    'Symptoms': "The device is running slow, system settings are modified without user's permission, questionable applications appear, data and battery usage is increased significantly, intrusive advertisements are delivered.",
                                    'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, deceptive applications, scam websites.',
                                    'Damage': 'Stolen personal information (private messages, logins/passwords, etc.), decreased device performance, battery is drained quickly, decreased Internet speed, huge data losses, monetary losses, stolen identity (malicious apps might abuse communication apps).'},
         'BianLian': {'Name': 'BianLian virus',
                      'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                      'Encrypted Files Extension': '.bianlian',
                      'Ransom Demanding Message': 'Look at this instruction.txt',
                      'Free Decryptor Available?': 'Yes (more information below)',
                      'Cyber Criminal Contact': 'Tox chat, swikipedia@onionmail.org',
                      'Detection Names': 'Avast (FileRepMalware [Ransom]), Combo Cleaner (Trojan.GenericKD.61254969), ESET-NOD32 (WinGo/Filecoder.BT), Kaspersky (Trojan-PSW.Win32.Stealer.aosa), Microsoft (Ransom:Win64/Bianlian!MSR), Full List Of Detections (VirusTotal)',
                      'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                      'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                      'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
         'Lilith': {'Name': 'Lilith virus',
                    'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                    'Encrypted Files Extension': '.lilith',
                    'Ransom Demanding Message': 'Restore_Your_Files.txt',
                    'Free Decryptor Available?': 'No',
                    'Cyber Criminal Contact': 'Tox chat, website on Tor network',
                    'Detection Names': 'Avast (Win64:Malware-gen), Combo Cleaner (Trojan.GenericKD.49307970), ESET-NOD32 (A Variant Of Win64/Filecoder.FC), Kaspersky (Trojan-Ransom.Win32.Encoder.rqk), Microsoft (Trojan:Win64/Vigorf.A), Full List Of Detections (VirusTotal)',
                    'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                    'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                    'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
         'LimeRAT': {'Name': 'LimeRat remote access tool',
                      'Threat Type': 'Remote access trojan',
                      'Detection Names (YfRrW5qu.exe)': 'Avast (Win32:MalwareX-gen [Trj]), BitDefender (Gen:Variant.Razy.396392), ESET-NOD32 (A Variant Of MSIL/Agent.BPK), Kaspersky (HEUR:Trojan.MSIL.Tasker.gen), Full List (VirusTotal)',
                      'Payload': 'Crytocurrency miner, worm, keystroke logger, screen grabber, information stealer, ransomware',
                      'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                      'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                      'Damage': "Stolen banking information, passwords, identity theft, victim's computer added to a botnet."},
         'CrimsonRat': {'Name': 'Trojan.CrimsonRat',
                          'Threat Type': 'Trojan, Password stealing virus, Banking malware, Spyware',
                          'Detection Names': 'Avast (Win32:TrojanX-gen [Trj]), BitDefenderTheta (Gen:NN.ZemsilF.34608.@p0@a4gDjW), ESET-NOD32 (A Variant Of MSIL/Agent.BNY), Kaspersky (HEUR:Trojan-Ransom.MSIL.Foreign.gen), Microsoft (Program:Win32/Wacapew.C!ml), Full List Of Detections (VirusTotal)',
                          'Symptoms': "Trojans are designed to stealthily infiltrate victim's computer and remain silent thus no particular symptoms are clearly visible on an infected machine.",
                          'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, software cracks.',
                          'Damage': "Stolen banking information, passwords, identity theft, victim's computer added to a botnet."},
         'ObliqueRAT': {'Name': 'ObliqueRAT virus',
                        'Threat Type': 'Remote Access Trojan',
                        'Detection Names': 'Avast (Win32:Trojan-gen), BitDefender (Trojan.GenericKD.45754215), ESET-NOD32 (Win32/Agent.ACQR), Kaspersky (UDS:DangerousObject.Multi.Generic), Microsoft (Trojan:Win32/CryptInject!MSR), Full List (VirusTotal)',
                        'Symptoms': "Remote Administration Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                        'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                        'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'CapraRAT': {'Name': 'CapraRAT remote access trojan',
                      'Threat Type': 'Android malware, malicious application, unwanted application.',
                      'Detection Names': 'Avast-Mobile (Android:Evo-gen [Trj]), Combo Cleaner (Trojan.GenericKD.49195185), ESET-NOD32 (A Variant Of Android/Spy.AndroRAT.AE), Kaspersky (HEUR:Trojan-Spy.AndroidOS.Camvod.a), Full List (VirusTotal)',
                      'Symptoms': "The device is running slow, system settings are modified without user's permission, questionable applications appear, data and battery usage is increased significantly, browsers redirect to questionable websites, intrusive advertisements are delivered.",
                      'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, deceptive applications, scam websites.',
                      'Damage': 'Stolen personal information (private messages, logins/passwords, etc.), decreased device performance, battery is drained quickly, decreased Internet speed, huge data losses, monetary losses, stolen identity (malicious apps might abuse communication apps).'},
         'Sality': {'Name': 'Sality trojan malware',
                    'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                    'Detection Names': 'Avast (Win32:Kukacka), BitDefender (Win32.Sality.OG), ESET-NOD32 (Win32/Sality.NAR), Kaspersky (Virus.Win32.Sality.gen), Full List (VirusTotal)',
                    'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                    'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                    'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'Keona Clipper': {'Name': 'Keona clipboard hijacker',
                              'Threat Type': 'Trojan, Clipper, Clipboard Hijacker.',
                              'Detection Names': 'Avast (Win64:BankerX-gen [Trj]), Combo Cleaner (Trojan.GenericKD.39722212), ESET-NOD32 (A Variant Of MSIL/ClipBanker.ABF), Kaspersky (HEUR:Trojan-Banker.MSIL.ClipBanker.gen), Microsoft (Trojan:Win32/Tiggre!rfn), Full List Of Detections (VirusTotal)',
                              'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                              'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                              'Damage': 'Stolen cryptocurrency, financial loss.'},
         'Snake Keylogger': {'Name': 'Snake malware',
                              'Threat Type': 'Keylogger, password-stealing virus, banking malware, spyware.',
                              'Detection Names': 'Avast (Win32:PWSX-gen [Trj]), BitDefenderTheta (Gen:NN.ZemsilF.34670.ym0@aq!9ljli), ESET-NOD32 (A Variant Of MSIL/Spy.Agent.AES), Kaspersky (HEUR:Trojan-Spy.MSIL.Stealer.gen), Microsoft (Trojan:Win32/Meterpreter!ml), Full List Of Detections (VirusTotal)',
                              'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                              'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                              'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'Raccoon Stealer': {'Name': 'Raccoon Stealer trojan',
                            'Threat Type': 'Trojan, Password-stealing virus, Banking malware, Spyware',
                            'Detection Names (2.exe)': 'Avast (Win32:Trojan-gen), BitDefender (Gen:Heur.Titirez.1.F), ESET-NOD32 (Win32/Spy.Agent.PQZ), Kaspersky (Trojan-Spy.MSIL.Stealer.aik), Full List (VirusTotal)',
                              'Malicious Process Name(s)': '2.exe (the name may vary).',
                              'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                              'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                              'Damage': "Stolen banking information, passwords, identity theft, victim's computer added to a botnet."},
        'PINEFLOWER': {'Name': 'PINEFLOWER virus',
                        'Threat Type': 'Android malware, malicious application.',
                        'Detection Names': 'Avast-Mobile (APK:RepMalware [Trj]), ESET-NOD32 (Android/Spy.Agent.AUR), Ikarus (Trojan.SuspectCRC), Kaspersky (UDS:DangerousObject.Multi.Generic), Full List (VirusTotal)',
                        'Symptoms': "The device is running slow, system settings are modified without user's permission, questionable applications appear, data and battery usage is increased significantly.",
                        'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, deceptive applications, scam websites.',
                        'Damage': 'Stolen personal information (private messages, logins/passwords, etc.), decreased device performance, battery is drained quickly, decreased Internet speed, huge data losses, monetary losses, stolen identity (malicious apps might abuse communication apps).'},
         'BlackByte': {'Name': 'BlackByte virus',
                      'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                      'Encrypted Files Extension': '.blackbyte',
                      'Ransom Demanding Message': 'BlackByte_restoremyfiles.hta',
                      'Free Decryptor Available?': 'Yes (more information below).',
                      'Cyber Criminal Contact': 'blackbyte1@onionmail.org',
                      'Detection Names': 'N/A',
                      'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                      'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                      'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
         'CloudMensis': {'Name': 'CloudMensis macOS malware',
                          'Threat Type': 'Spyware',
                          'Detection Names': 'Avast (MacOS:Imis-A [Trj]), Combo Cleaner (Trojan.MAC.Generic.109834), ESET-NOD32 (OSX/CloudMensis.A), Kaspersky (HEUR:Trojan-Spy.OSX.Agent.d), Full List (VirusTotal)',
                          'Symptoms': "Spyware is designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                          'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                          'Damage': 'Stolen passwords and banking information, identity theft, additional infections, loss of access to pesonal accounts, monetary loss.'},
         'Manjusaka': {'Name': 'Manjusaka framework',
                        'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                        'Detection Names (Malicious Word Document)': 'Avast (VBS:HackTool-B [Trj]), Combo Cleaner (Trojan.GenericKD.61147495), ESET-NOD32 (VBA/TrojanDownloader.Agent.WOE), Kaspersky (UDS:DangerousObject.Multi.Generic), Microsoft (Trojan:X97M/LionWolf.A), Full List (VirusTotal)',
                        'Detection Names (Manjusaka)': 'Avast (Win64:TrojanX-gen [Trj]), Combo Cleaner (Trojan.GenericKD.61133964), ESET-NOD32 (A Variant Of Win64/Manjusaka.A), Kaspersky (Trojan.Win32.Agent.xaqdfo), Microsoft (Trojan:Win64/Tnega!MSR), Full List (VirusTotal)',
                        'Symptoms': "This malware stealthily infiltrates the victim's computer and remains silent, and thus no particular symptoms are clearly visible on an infected machine.",
                        'Distribution methods': 'Infected MS Word documents.',
                        'Damage': 'Stolen passwords and banking information, hijacked online accounts, identity theft, data loss, and more.'},
         'Amadey': {'Name': 'Amadey bot',
                      'Threat Type': 'Trojan, Botnet, Password-stealing virus, Banking malware, Spyware, Keylogger.',
                      'Detection Names': 'Avast (Win32:Malware-gen), BitDefender (Trojan.GenericKD.31664374), ESET-NOD32 (Win32/TrojanDownloader.Agent.EGF), Kaspersky (Trojan-Dropper.Win32.Dapato.prmr), Full List (VirusTotal)',
                      'Payload': 'Amadey can be used to install other malware such as ransomware, Trojans, and so on.',
                      'Symptoms': 'Inability to start the computer in Safe Mode, open Registry Editor or Task Manager, increased disk and network activity.',
                      'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, software cracks.',
                      'Damage': "Stolen banking information, passwords, identity theft, victim's computer added to a botnet, installation of additional malware, victims computer used to send spam to other people."},
         'DevilsTongue': {'Name': 'DevilsTongue virus',
                          'Threat Type': 'Trojan, password-stealing virus, banking malware, spyware.',
                          'Detection Names': 'AhnLab-V3 (Unwanted/Win.DevilsTongue.C4553761), Webroot (W32.Malware.Devilstongue), Full List Of Detections (VirusTotal)',
                          'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                          'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                          'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'Solidbit': {'Name': 'Solidbit virus',
                      'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                      'Encrypted Files Extension': 'Randomly generated extension (four characters), .solidbit (depends on the variant)',
                      'Ransom Demanding Message': 'RESTORE-MY-FILES.txt, pop-up window',
                      'Free Decryptor Available?': 'No',
                      'Detection Names': 'Avast (Win32:Trojan-gen), Combo Cleaner (Gen:Heur.Ransom.REntS.Gen.1), ESET-NOD32 (A Variant Of MSIL/Filecoder.APU), Kaspersky (HEUR:Trojan-Ransom.MSIL.Encoder.gen), Microsoft (Ransom:MSIL/Filecoder.PK!MSR), Full List Of Detections (VirusTotal)',
                      'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                      'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                      'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
        'BRATA': {'Name': 'BRATA remote access trojan',
                  'Threat Type': 'Android malware, malicious application, unwanted application.',
                  'Detection Names': 'Avast-Mobile (Android:Evo-gen [Trj]), BitDefenderFalx (Android.Trojan.Banker.UQ), ESET-NOD32 (Android/Agent.CBO), Kaspersky (HEUR:Trojan.AndroidOS.Piom.agmk), Full List (VirusTotal)',
                  'Symptoms': "The device is running slow, system settings are modified without user's permission, questionable applications appear, data and battery usage is increased significantly, browsers redirect to questionable websites, intrusive advertisements are delivered.",
                  'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, deceptive applications, scam websites.',
                  'Damage': 'Stolen personal information (private messages, logins/passwords, etc.), decreased device performance, battery is drained quickly, decreased Internet speed, huge data losses, monetary losses, stolen identity (malicious apps might abuse communication apps).'},
         'Diavol': {'Name': 'Diavol virus',
                      'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                      'Encrypted Files Extension': '.lock64',
                      'Ransom Demanding Message': 'README_FOR_DECRYPT.txt',
                      'Free Decryptor Available?': 'Yes (it can be downloadedhere)',
                      'Cyber Criminal Contact': 'Tor website',
                      'Detection Names': 'Avast (Win64:Trojan-gen), Combo Cleaner (Trojan.GenericKD.47184519), ESET-NOD32 (A Variant Of Win64/Filecoder.Diavol.A), Kaspersky (Trojan-Ransom.Win32.Encoder.ocs), Microsoft (Trojan:Win32/Sabsik.FL.B!ml), Full List Of Detections (VirusTotal)',
                      'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                      'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                      'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
         'RomCom, ROMCOM RAT': {'Name': 'RomCom remote access trojan',
                                  'Threat Type': 'Trojan, RAT, Remote Access Trojan, password-stealing virus, banking malware, spyware.',
                                  'Detection Names': 'Avast (Win64:MalwareX-gen [Trj]), Combo Cleaner (Trojan.GenericKD.61083265), ESET-NOD32 (A Variant Of Win64/Agent.QU), Kaspersky (Trojan.Win64.Agentb.ktrn), Microsoft (Trojan:Win64/CobaltStrike!MTB), Full List Of Detections (VirusTotal)',
                                  'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                                  'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                                  'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
         'RansomExx': {'Name': 'RansomExx virus',
                      'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                      'Encrypted Files Extension': "Depends on the target's name",
                      'Ransom Demand Message': "Depends on the target's name",
                      'Cyber Criminal Contact': 'Email address for contacting the attackers makes use of the target’s name',
                      'Detection Names': 'Avast (Win32:Malware-gen), BitDefender (Gen:Heur.Ransom.REntS.Gen.1), ESET-NOD32 (A Variant Of Win32/Filecoder.OCN), Kaspersky (Trojan-Ransom.Win32.Ransomexx.e), Microsoft (Ransom:Win32/FileCoder.TX!MSR), Full List Of Detections (VirusTotal)',
                      'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                      'Additional Information': 'RansomExx targets not only Windows but also Linux systems',
                      'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                      'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing Trojans and malware infections can be installed together with a ransomware infection.'},
         'S.O.V.A., SOVA': {'Name': 'S.O.V.A. malware',
                              'Threat Type': 'Android malware, malicious application, unwanted application.',
                              'Detection Names': 'Avast-Mobile (Android:Evo-gen [Trj]), ESET-NOD32 (A Variant Of Generik.GLAMSH), DrWeb (Android.BankBot.842.origin), Kaspersky (HEUR:Trojan-Banker.AndroidOS.Sova.a), Full List (VirusTotal)',
                              'Symptoms': "The device is running slow, system settings are modified without user's permission, questionable applications appear, data and battery usage is increased significantly, browsers redirect to questionable websites, intrusive advertisements are delivered.",
                              'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, deceptive applications, scam websites.',
                              'Damage': 'Stolen personal information (private messages, logins/passwords, etc.), decreased device performance, battery is drained quickly, decreased Internet speed, huge data losses, monetary losses, stolen identity (malicious apps might abuse communication apps).'},
          'Duke, TheDukes': {'Name': 'Duke malware toolset',
                              'Threat Type': 'Trojan, backdoor, loader, password-stealing virus, banking malware, spyware.',
                              'Detection Names (PDF attachment)': 'Avast (Other:Malware-gen [Trj]), Combo Cleaner (Trojan.GenericKD.68283668), ESET-NOD32 (PDF/TrojanDropper.Agent.CJ), Kaspersky (Trojan-Dropper.HTML.Agent.cn), Microsoft (Trojan:Win32/Leonem), Full List Of Detections (VirusTotal)',
                              'Detection Names (PDF payload)': 'Avast (Win64:MalwareX-gen [Trj]), Combo Cleaner (Trojan.GenericKD.68275714), ESET-NOD32 (A Variant Of Win64/Dukes.N), Kaspersky (Trojan.Win64.Agent.qwikue), Microsoft (Trojan:Win32/Casdet!rfn), Full List Of Detections (VirusTotal)',
                              'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                              'Distribution methods': "Infected email attachments, malicious online advertisements, social engineering, software 'cracks'.",
                              'Damage': "Stolen passwords and banking information, identity theft, the victim's computer added to a botnet."},
             'Pegasus': {'Name': 'Pegasus virus',
                                'Threat Type': 'Ransomware, Crypto Virus, Files locker',
                                'Encrypted Files Extension': 'Files are appended with an extension comprising a random character string',
                                'Ransom Demanding Message': 'Ghost_ReadMe.txt',
                                'Ransom Amount': '$350 in Bitcoin cryptocurrency',
                                'Cyber Criminal Cryptowallet Address': '16JpyqQJ6z1GbxJNztjUnepXsqee3SBz75 (Bitcoin)',
                                'Free Decryptor Available?': 'No',
                                'Cyber Criminal Contact': 'ransom.data@gmail.com',
                                'Detection Names': 'Avast (Win32:RansomX-gen [Ransom]), Combo Cleaner (Generic.Ransom.Hiddentear.A.29961126), ESET-NOD32 (A Variant Of MSIL/Filecoder.AXL), Kaspersky (HEUR:Trojan-Ransom.Win32.Generic), Microsoft (Ransom:MSIL/HiddenTear.RDA!MTB), Full List Of Detections (VirusTotal)',
                                'Symptoms': 'Cannot open files stored on your computer, previously functional files now have a different extension (for example, my.docx.locked). A ransom demand message is displayed on your desktop. Cyber criminals demand payment of a ransom (usually in bitcoins) to unlock your files.',
                                'Distribution methods': 'Infected email attachments (macros), torrent websites, malicious ads.',
                                'Damage': 'All files are encrypted and cannot be opened without paying a ransom. Additional password-stealing trojans and malware infections can be installed together with a ransomware infection.'},
             'WarzoneRAT': {'Name': 'Warzone remote access trojan',
                                              'Threat Type': 'Trojan, Password-stealing virus, Banking malware, Spyware',
                                              'Detection Names': 'Avast (Win32:Malware-gen), BitDefender (Gen:Variant.Graftor.527299), ESET-NOD32 (A Variant Of Win32/Agent.TJS), Kaspersky (Trojan.Win32.Agentb.jiad), Full List (VirusTotal)',
                                              'Symptoms': "Trojans are designed to stealthily infiltrate the victim's computer and remain silent, and thus no particular symptoms are clearly visible on an infected machine.",
                                              'Distribution methods': 'Infected email attachments, malicious online advertisements, social engineering, software cracks.',
                                              'Damage': "Stolen banking information, passwords, identity theft, victim's computer added to a botnet."}}

def get_total_describe_of_malware(text):
    malwares = get_malware(text)
    total_describe = []
    for key in total_dict:
          for malware in malwares:
            if malware in key:
                dct = total_dict.get(key)
            
                for key in dct:
                    total_describe.append(f"{key}: {dct.get(key)}")
                    
    total_describe = list(OrderedDict.fromkeys(total_describe))
    
    return total_describe

st.title("Сервис по суммаризации и извлечению метрик из новостных статей")

url_input = st.text_input("Введите ссылку на новость:")

if st.button("Прочитать новость"):
    
    if url_input:
        news_text = fetch_news_text(url_input)
        
        st.text_area(label = 'Текст новости:', value = news_text, height=200)
        st.markdown("<h1 style='text-align: center; font-size: 24px; '>Метрики:</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
            
        col1.text_area(label = 'ВПО:', value = ', '.join([i for i in get_malware(news_text)]))

        col1.text_area(label = 'Тип ВПО:', value = ', '.join([i for i in get_type_vpo(get_malware(news_text))]))

        col1.text_area(label = 'Группировка:', value = ', '.join([i for i in get_group(get_malware(news_text))]))

        col1.text_area(label = 'Дата:', value = ', '.join([i for i in get_date(news_text)]))

        col1.text_area(label = 'Сумма ущерба:', value = ', '.join([i for i in get_monetary_damage(news_text)]))

        col1.text_area(label = 'Жертва:', value = ', '.join([i for i in get_victim(news_text)]))

        col1.text_area(label = 'Категория жертвы:', value = get_category_of_victim(get_victim(news_text)))

        col2.text_area(label = 'Описание:', value = get_description(news_text), height = 245)

        col2.text_area(label = 'Атакующий:', value = ', '.join([i for i in get_attacker(news_text)]))

        col2.text_area(label = 'Последствия атаки:', value = get_consequences(news_text))

        col2.text_area(label = 'Метод атаки:', value = get_method_of_attack(news_text))

        col2.text_area(label = 'Тип атаки:', value = get_target_mass_attack(news_text))

        col2.text_area(label = 'Уязвимости:', value = ', '.join(i for i in cve_format(news_text)))

        st.text_area(label = 'Краткое описание ВПО:', value = '\n\n'.join([i for i in description_of_malware(news_text)]), height = 200)

        st.text_area(label = 'Детальное описание ВПО:', value = '\n\n'.join([i for i in get_total_describe_of_malware(news_text)]), height = 730)
        
    else:
        st.error("Пожалуйста, введите ссылку на новость.")