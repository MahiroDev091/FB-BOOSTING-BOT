import requests
from fbchat import Client
from fbchat.models import *
import ua_generator
import re
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import json
import sys
import time

try:
				with open('configuration.json') as f:
					configuration = json.load(f)
except FileNotFoundError:
			print("\033[1m\033[91mSORRY, AN ERROR ENCOUNTERED WHILE FINDING 'CONFIGURATION.JSON'.\033[0m")
			sys.exit()
except json.decoder.JSONDecodeError:
			print("\033[1m\033[91mSORRY, AN ERROR ENCOUNTERED WHILE READING THE JSON FILE.\033[0m")
			sys.exit()

def print_slow(str):
            for char in str:
            	time.sleep(.1)
            	sys.stdout.write(char)
            	sys.stdout.flush()
            sys.exit()


class MessBot(Client):
    add_token = []

    def get_token(self):
        global configuration
        os.system('clear')
        accounts = configuration['CONFIG']['PAGE_ACCOUNTS']['ACCOUNTS']
        for account in accounts:
            account_data = account.split('|')
            url = 'https://b-api.facebook.com/method/auth.login'
            form = {
                'adid': 'e3a395f9-84b6-44f6-a0ce-fe83e934fd4d',
                'email': account_data[0],
                'password': account_data[1],
                'format': 'json',
                'device_id': '67f431b8-640b-4f73-a077-acc5d3125b21',
                'cpl': 'true',
                'family_device_id': '67f431b8-640b-4f73-a077-acc5d3125b21',
                'locale': 'en_US',
                'client_country_code': 'US',
                'credentials_type': 'device_based_login_password',
                'generate_session_cookies': '1',
                'generate_analytics_claim': '1',
                'generate_machine_id': '1',
                'currently_logged_in_userid': '0',
                'irisSeqID': 1,
                'try_num': '1',
                'enroll_misauth': 'false',
                'meta_inf_fbmeta': 'NO_FILE',
                'source': 'login',
                'machine_id': 'KBz5fEj0GAvVAhtufg3nMDYG',
                'meta_inf_fbmeta': '',
                'fb_api_req_friendly_name': 'authenticate',
                'fb_api_caller_class': 'com.facebook.account.login.protocol.Fb4aAuthHandler',
                'api_key': '882a8490361da98702bf97a021ddc14d',
                'access_token': '350685531728%7C62f8ce9f74b12f84c123cc23437a4a32'
            }
            headers = {
                'content-type': 'application/x-www-form-urlencoded',
                'x-fb-friendly-name': 'fb_api_req_friendly_name',
                'x-fb-http-engine': 'Liger',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
            }
            response = requests.post(url, data=form, headers=headers)
            try:
                response_data = response.json()
                if 'access_token' in response_data:
                    self.add_token.append(response_data['access_token'])
                    print("\033[1m[\033[91m\033[1m/\033[0m\033[1m] PAGES SUCCESSFULLY LOADED!\033[0m")
                else:
                    print("\033[1m[\033[91m\033[1mx\033[0m\033[1m] PAGES FAILED TO LOAD!\033[0m")
            except ValueError as e:
                print("\033[1m[\033[91m\033[1mx\033[0m\033[1m] Error decoding JSON for {} {}: {}\033[0m".format(
                    account_data[0], account_data[1], e))

    def react(self, post_id, reaction_type):
        for access_token in self.add_token:
            try:
                response = requests.get(
                    f'https://graph.facebook.com/me/accounts?access_token={access_token}').json()

                for page in response.get('data', []):
                    page_access_token = page.get('access_token', '')
                    page_name = page.get('name', '')

                    try:
                        headers = {
                            'content-type': 'application/x-www-form-urlencoded',
                            'x-fb-friendly-name': 'fb_api_req_friendly_name',
                            'x-fb-http-engine': 'Liger',
                            'user-agent': str(ua_generator.generate())
                        }
                        response = requests.get(f'https://mahirochan.pythonanywhere.com/api', params={'reaction_type': reaction_type.upper(), 'link': post_id, 'access_token': page_access_token}, headers=headers)
                        if response.status_code == 200:
                            print("\033[0m\033[1m[\033[91mSUCCESS\033[0m\033[1m] SUCCESSFULLY REACTION |\033[91m {}\033[0m \033[1m|\033[90m {}\033[0m".format(
                                page_name, str(response.json())))
                        else:
                            print("\033[1;91m[ERROR]\033[0;1m FAILED TO POST REACTION \033[0m")
                            pass
                    except requests.exceptions.RequestException as error:
                        print("\033[1;91m[EXCEPTION]\033[0;1m {}\033[0m".format(error))
            except requests.exceptions.RequestException as error:
                print("\033[1;91m[EXCEPTION]\033[0m {}".format(error))

    def follow(self, account_id):
        for token in self.add_token:
            success_followed = False
            headers = {'Authorization': f'Bearer {token}'}

            scope = [
                'public_profile', 'email', 'user_friends', 'user_likes', 'user_photos',
                'user_videos', 'user_status', 'user_posts', 'user_tagged_places', 'user_hometown',
                'user_location', 'user_work_history', 'user_education_history', 'user_groups',
                'publish_pages', 'manage_pages'
            ]
            data = {'scope': ','.join(scope)}

            response = requests.get(
                'https://graph.facebook.com/v18.0/me/accounts', headers=headers, params=data)
            pages_data = response.json().get('data', [])

            for page in pages_data:
                page_access_token = page.get('access_token', '')
                page_name = page.get('name', '')

                try:
                    response = requests.post(
                        f'https://graph.facebook.com/v18.0/{account_id}/subscribers', headers={'Authorization': f'Bearer {page_access_token}'})
                    print("\033[0m\033[1m[\033[91mSUCCESS\033[0m\033[1m] SUCCESSFULLY FOLLOW |\033[91m {}\033[0m \033[1m|\033[91m {}\033[0m \033[1m|\033[91m {}\033[0m".format(
                        page_name, account_id, response))
                    success_followed = True
                except requests.exceptions.RequestException as error:
                    print(error)

    def sendmessage(self, author_id, thread_id, thread_type, reply):
        if author_id != self.uid:
            self.send(Message(text=reply),
                      thread_id=thread_id,
                      thread_type=thread_type)

    def onMessage(self, mid=None, author_id=None, message_object=None, thread_id=None, thread_type=ThreadType.USER, **kwargs):
        try:
            global follow_in_progress, reaction_in_progress
            with open('configuration.json') as f:
            	configuration = json.load(f)
            msg = message_object.text.lower()
            rainbow_light_text_print("[ [ MESSAGE ] ] " + msg)
            prefix = str(configuration['CONFIG']['BOT_INFO']['PREFIX'])
            prefixs = ("prefix", "PREFIX", "Mahiro", "MAHIRO", "Prefix")
            if any(msg.startswith(prefix) for prefix in prefixs):
            	reply = f"𝙷𝙾𝚆 𝚃𝙾 𝚄𝚂𝙴:\n- {prefix}𝚏𝚋𝚏𝚘𝚕𝚕𝚘𝚠 [𝚒𝚍]\n- {prefix}𝚏𝚋𝚛𝚎𝚊𝚌𝚝 [𝙻𝙸𝙺𝙴/𝙻𝙾𝚅𝙴/𝚂𝙰𝙳/𝙰𝙽𝙶𝚁𝚈/𝙷𝙰𝙷𝙰] [𝚕𝚒𝚗𝚔]\n\n𝙼𝚊𝚔𝚎 𝚜𝚞𝚛𝚎 𝚝𝚑𝚊𝚝 𝚝𝚑𝚎 𝚕𝚒𝚗𝚔 𝚢𝚘𝚞'𝚛𝚎 𝚞𝚜𝚒𝚗𝚐 𝚒𝚜 𝚏𝚛𝚘𝚖 𝚏𝚋𝚕𝚒𝚝𝚎 𝚘𝚛 𝚎𝚕𝚜𝚎 𝚒𝚝 𝚖𝚒𝚐𝚑𝚝 𝚗𝚘𝚝 𝚠𝚘𝚛𝚔.\n\n𝚃𝚢𝚙𝚎 '{prefix}𝚕𝚒𝚜𝚝' 𝚝𝚘 𝚜𝚑𝚘𝚠 𝚊𝚟𝚊𝚒𝚕𝚊𝚋𝚕𝚎 𝚌𝚘𝚖𝚖𝚊𝚗𝚍𝚜."
            	self.sendmessage(author_id, thread_id, thread_type, reply)
            dev = ("dev", "owner", "Owner", "Developer", "developer", "OWNER", "DEVELOPER", "DEV")
            if any(msg.startswith(word) for word in dev):
                reply = "𝙳𝙴𝚅𝙴𝙻𝙾𝙿𝙴𝚁: 𝙼𝙰𝙷𝙸𝚁𝙾 𝙲𝙷𝙰𝙽"
                self.sendmessage(author_id, thread_id, thread_type, reply)
            greetings = ("hi", "Hi", "hello", "Hello", "hi!", "Hi!", "hello!", "Hello!")
            if any(msg.startswith(greeting) for greeting in greetings):
                sender_name = self.fetchUserInfo(author_id)[author_id].name
                reply = f"Hello, {sender_name}!"
                self.sendmessage(author_id, thread_id, thread_type, reply)
            if msg.startswith(f"{prefix}fbfollow"):
                if follow_in_progress:
                    reply = "⌛𝚃𝚑𝚎 𝚜𝚕𝚘𝚝 𝚒𝚜 𝚌𝚞𝚛𝚛𝚎𝚗𝚝𝚕𝚢 𝚘𝚌𝚌𝚞𝚙𝚒𝚎𝚍; 𝚠𝚎 𝚊𝚕𝚕𝚘𝚠 𝚘𝚗𝚕𝚢 𝚘𝚗𝚎 𝚛𝚎𝚚𝚞𝚎𝚜𝚝 𝚊𝚝 𝚊 𝚝𝚒𝚖𝚎. 𝙿𝚕𝚎𝚊𝚜𝚎 𝚛𝚎𝚝𝚛𝚢 𝚊𝚏𝚝𝚎𝚛 𝚊 𝚏𝚎𝚠 𝚖𝚒𝚗𝚞𝚝𝚎𝚜."
                    self.sendmessage(author_id, thread_id, thread_type, reply)
                else:
                    follow_in_progress = True
                    id = msg[len(prefix)+9:]
                    allow = ['100', '615']

                    if "https://www.facebook.com/" in id or "https://m.facebook.com/story.php" in id or not any(id.startswith(allowed) for allowed in allow):
                        reply = "❌ 𝙸𝙳 𝙽𝙾𝚃 𝙵𝙾𝚄𝙽𝙳!"
                        self.sendmessage(author_id, thread_id, thread_type, reply)
                        follow_in_progress = False
                    else:
                        reply = "⌛𝙿𝚁𝙾𝙲𝙴𝚂𝚂𝙸𝙽𝙶 𝙿𝚄𝚁𝙲𝙷𝙰𝚂𝙴, 𝙿𝙻𝙴𝙰𝚂𝙴 𝚆𝙰𝙸𝚃.."
                        self.sendmessage(author_id, thread_id, thread_type, reply)

                        def f():
                            try:
                                self.get_token()
                                self.follow(id)
                            except Exception as e:
                                return str(e)
                            finally:
                                global follow_in_progress
                                follow_in_progress = False
                                reply = "🗒𝙾𝚁𝙳𝙴𝚁 𝚂𝚄𝙲𝙲𝙴𝚂𝚂𝙵𝚄𝙻𝙻𝚈 𝙰𝚁𝚁𝙸𝚅𝙴𝙳."
                                self.sendmessage(author_id, thread_id, thread_type, reply)

                        success_followed = threading.Thread(target=f)
                        success_followed.start()

                        if success_followed:
                            reply = f"[ 𝙿𝚄𝙲𝙷𝙰𝚂𝙴 𝚂𝚄𝙲𝙲𝙴𝚂𝚂𝙵𝚄𝙻𝙻𝚈 𝚂𝙴𝙽𝚃 ]\n🔗𝙿𝚁𝙾𝙵𝙸𝙻𝙴 𝙻𝙸𝙽𝙺: https://www.facebook.com/{id}\n💢𝙼𝙴𝚃𝙷𝙾𝙳 𝚂𝙴𝚁𝚅𝙸𝙲𝙴: 𝙵𝙱𝙵𝙾𝙻𝙻𝙾𝚆\n🏆𝙿𝚁𝙴𝙼𝙸𝚄𝙼: 𝙽𝙾(𝙵𝚁𝙴𝙴)\n\n[+] ᴅᴇᴠᴇʟᴏᴘᴇᴅ ʙʏ ᴍᴀʜɪʀᴏ ᴄʜᴀɴ"
                            self.sendmessage(author_id, thread_id, thread_type, reply)
                        else:
                            reply = "❌𝙵𝙰𝙸𝙻𝙴𝙳 𝙿𝚄𝚁𝙲𝙷𝙰𝚂𝙴 𝚁𝙴𝚀𝚄𝙴𝚂𝚃."
                            self.sendmessage(author_id, thread_id, thread_type, reply)
            if msg.startswith(f"{prefix}fbreact"):
                if reaction_in_progress:
                    reply = "⌛𝚃𝚑𝚎 𝚜𝚕𝚘𝚝 𝚒𝚜 𝚌𝚞𝚛𝚛𝚎𝚗𝚝𝚕𝚢 𝚘𝚌𝚌𝚞𝚙𝚒𝚎𝚍; 𝚠𝚎 𝚊𝚕𝚕𝚘𝚠 𝚘𝚗𝚕𝚢 𝚘𝚗𝚎 𝚛𝚎𝚚𝚞𝚎𝚜𝚝 𝚊𝚝 𝚊 𝚝𝚒𝚖𝚎. 𝙿𝚕𝚎𝚊𝚜𝚎 𝚛𝚎𝚝𝚛𝚢 𝚊𝚏𝚝𝚎𝚛 𝚊 𝚏𝚎𝚠 𝚖𝚒𝚗𝚞𝚝𝚎𝚜."
                    self.sendmessage(author_id, thread_id, thread_type, reply)
                else:
                    reaction_in_progress = True
                    id_and_link = msg[len(prefix)+8:].split(" ")
                    if len(id_and_link) < 2:
                        reply = "❌𝚆𝚁𝙾𝙽𝙶 𝙵𝙾𝚁𝙼𝙰𝚃!"
                        self.sendmessage(author_id, thread_id, thread_type, reply)
                        reaction_in_progress = False
                    else:
                        reaction_in_progress = True
                        me = msg[len(prefix)+8:].split(" ")
                        id = me[0].upper()
                        link = me[1]
                        if id not in ['LIKE', 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY'] or "https://www.facebook.com/" not in link:
                            reply = "❌𝚆𝚁𝙾𝙽𝙶 𝚁𝙴𝙰𝙲𝚃𝙸𝙾𝙽 𝚃𝚈𝙿𝙴 𝙾𝚁 𝚆𝚁𝙾𝙽𝙶 𝚄𝚁𝙻"
                            self.sendmessage(author_id, thread_id, thread_type, reply)
                            reaction_in_progress = False
                        else:
                            reply = "⌛𝙿𝚁𝙾𝙲𝙴𝚂𝚂𝙸𝙽𝙶 𝙿𝚄𝚁𝙲𝙷𝙰𝚂𝙴, 𝙿𝙻𝙴𝙰𝚂𝙴 𝚆𝙰𝙸𝚃.."
                            self.sendmessage(author_id, thread_id, thread_type, reply)

                            def r():
                                try:
                                    self.get_token()
                                    self.react(link, id)
                                except Exception as e:
                                    return str(e)
                                finally:
                                	global reaction_in_progress
                                	reaction_in_progress = False
                                	reply = "🗒𝙾𝚁𝙳𝙴𝚁 𝚂𝚄𝙲𝙲𝙴𝚂𝚂𝙵𝚄𝙻𝙻𝚈 𝙰𝚁𝚁𝙸𝚅𝙴𝙳."
                                	self.sendmessage(author_id, thread_id, thread_type, reply)

                            success_reaction = threading.Thread(target=r)
                            success_reaction.start()
                            if success_reaction:
                                    reply = f"[ 𝙿𝚄𝙲𝙷𝙰𝚂𝙴 𝚂𝚄𝙲𝙲𝙴𝚂𝚂𝙵𝚄𝙻𝙻𝚈 𝚂𝙴𝙽𝚃 ]\n🔗𝙿𝚁𝙾𝙵𝙸𝙻𝙴 𝙻𝙸𝙽𝙺: {link}\n💢𝙼𝙴𝚃𝙷𝙾𝙳 𝚂𝙴𝚁𝚅𝙸𝙲𝙴: 𝙵𝙱𝚁𝙴𝙰𝙲𝚃𝙸𝙾𝙽\n🏆𝙿𝚁𝙴𝙼𝙸𝚄𝙼: 𝙽𝙾(𝙵𝚁𝙴𝙴)\n\n[+] ᴅᴇᴠᴇʟᴏᴘᴇᴅ ʙʏ ᴍᴀʜɪʀᴏ ᴄʜᴀɴ"
                                    self.sendmessage(author_id, thread_id, thread_type, reply)
                            else:
                                    reply = "❌𝙵𝙰𝙸𝙻𝙴𝙳 𝙿𝚄𝚁𝙲𝙷𝙰𝚂𝙴 𝚁𝙴𝚀𝚄𝙴𝚂𝚃."
                                    self.sendmessage(author_id, thread_id, thread_type, reply)
            if msg.startswith(f"{prefix}setprefix"):
            	if author_id in configuration['CONFIG']['BOT_INFO']['ADMIN_ID']:
            		new = msg[len(prefix)+10:]
            		if new == "" or " " in new or len(new) != 1:
            			reply = "❌𝙿𝚁𝙴𝙵𝙸𝚇 𝙼𝚄𝚂𝚃 𝙷𝙰𝚅𝙴 𝚅𝙰𝙻𝚄𝙴 𝙰𝙽𝙳 𝙳𝙾𝙴𝚂𝙽'𝚃 𝙷𝙰𝚅𝙴 𝚂𝙿𝙰𝙲𝙴 𝙰𝙽𝙳 𝙾𝙽𝙻𝚈 𝙾𝙽𝙴 𝚂𝚈𝙼𝙱𝙾𝙻/𝙻𝙴𝚃𝚃𝙴𝚁."
            			self.sendmessage(author_id, thread_id, thread_type, reply)
            		else:
            			with open("configuration.json", "r") as jsonFile:
            				data = json.load(jsonFile)
            			data['CONFIG']['BOT_INFO']['PREFIX'] = str(new)
            			with open("configuration.json", "w") as jsonFile:
            				json.dump(data, jsonFile, indent=3)
            			reply = f"✅𝙿𝚁𝙴𝙵𝙸𝚇 𝚆𝙰𝚂 𝚂𝚄𝙲𝙲𝙴𝚂𝚂𝙵𝚄𝙻𝙻𝚈 𝙲𝙷𝙰𝙽𝙶𝙴𝙳.\n𝙽𝙴𝚆 𝙿𝚁𝙴𝙵𝙸𝚇: {new}" 
            			self.sendmessage(author_id, thread_id, thread_type, reply)
            	else:
            		reply = "❌𝙾𝙽𝙻𝚈 𝙰𝙳𝙼𝙸𝙽 𝙲𝙰𝙽 𝙰𝙲𝙲𝙴𝚂𝚂 𝚃𝙷𝙸𝚂 𝙲𝙾𝙼𝙼𝙰𝙽𝙳."
            		self.sendmessage(author_id, thread_id, thread_type, reply)
            if msg.startswith(f"{prefix}catfact"):
            	facts = requests.get('https://catfact.ninja/fact').json()['fact']
            	reply = f"𝙲𝙰𝚃𝙵𝙰𝙲𝚃 𝚁𝙴𝚂𝙿𝙾𝙽𝙳: \n{facts}"
            	self.sendmessage(author_id, thread_id, thread_type, reply)
            if msg.startswith(f"{prefix}list"):
                reply = f"""[ 𝙁𝘽 𝘽𝙊𝙊𝙎𝙏𝙄𝙉𝙂 𝘽𝙊𝙏 ]
𝙳𝙴𝚅𝙴𝙻𝙾𝙿𝙴𝚁: 𝙼𝙰𝙷𝙸𝚁𝙾 𝙲𝙷𝙰𝙽
╭─❍
➠ {prefix}fbfollow: Send follow to id.
╰───────────⟡
╭─❍
➠ {prefix}fbreact: Send reaction to post.
╰───────────⟡
╭─❍
➠ {prefix}echo: say something.
╰───────────⟡
╭─❍
➠ {prefix}catfact: Get random catfacts everday.
╰───────────⟡
╭─❍
➠ {prefix}note: message from developer.
╰───────────⟡
╭─❍
➠ {prefix}uid: get your id.
╰───────────⟡
╭─❍
➠ {prefix}setprefix: change the prefix of bot [ADMIN ONLY].
╰───────────⟡"""
                self.sendmessage(author_id, thread_id, thread_type, reply)
            if msg.startswith(f"{prefix}uid"):
            	sender_name = self.fetchUserInfo(author_id)[author_id].name
            	reply = f"Hi, {sender_name}\n𝚃𝙷𝙸𝚂 𝙸𝚂 𝚈𝙾𝚄𝚁 𝙸𝙳:\n{author_id}"
            	self.sendmessage(author_id, thread_id, thread_type, reply)
            if msg.startswith(f"{prefix}ai"):
            	reply = "⌛𝙰𝙽𝚂𝚆𝙴𝚁𝙸𝙽𝙶 𝚈𝙾𝚄𝚁 𝚀𝚄𝙴𝚂𝚃𝙸𝙾𝙽, 𝙿𝙻𝙴𝙰𝚂𝙴 𝚆𝙰𝙸𝚃"
            	self.sendmessage(author_id, thread_id, thread_type, reply)
            	try:
            		ask = msg[len(prefix)+3:]
            		ask2 = requests.get('https://api.kenliejugarap.com/ai/?text=' + ask).json()['response']
            		reply = f"𝙰𝙸 𝚁𝙴𝚂𝙿𝙾𝙽𝙳: \n{ask2}"
            		self.sendmessage(author_id, thread_id, thread_type, reply)
            	except:
            		reply = "❌𝚂𝙾𝚁𝚁𝚈, 𝚆𝙴 𝙰𝚁𝙴 𝙷𝙰𝚅𝙸𝙽𝙶 𝙴𝚁𝚁𝙾𝚁 𝙵𝙴𝚃𝙲𝙷𝙸𝙽𝙶 𝚁𝙴𝚂𝙿𝙾𝙽𝙳."
            		self.sendmessage(author_id, thread_id, thread_type, reply)
            if msg.startswith(f"{prefix}note"):
            	reply = "𝚃𝚑𝚒𝚜 𝚋𝚘𝚝 𝚒𝚜 𝚌𝚞𝚛𝚛𝚎𝚗𝚝𝚕𝚢 𝚒𝚗 𝚊 𝚝𝚎𝚜𝚝 𝚙𝚑𝚊𝚜𝚎 𝚠𝚒𝚝𝚑 𝚏𝚛𝚎𝚎𝚖𝚘𝚍𝚎 𝚊𝚌𝚝𝚒𝚟𝚊𝚝𝚎𝚍. 𝙸𝚏 𝚢𝚘𝚞 𝚝𝚛𝚢 𝚛𝚎𝚊𝚌𝚝𝚒𝚗𝚐 𝚝𝚘 𝚝𝚑𝚎 𝚜𝚊𝚖𝚎 𝚙𝚘𝚜𝚝 𝚊 𝚜𝚎𝚌𝚘𝚗𝚍 𝚝𝚒𝚖𝚎, 𝚒𝚝 𝚖𝚒𝚐𝚑𝚝 𝚗𝚘𝚝 𝚠𝚘𝚛𝚔 𝚋𝚎𝚌𝚊𝚞𝚜𝚎 𝚝𝚑𝚎 𝚍𝚊𝚝𝚊𝚋𝚊𝚜𝚎 𝚑𝚊𝚜 𝚊𝚕𝚛𝚎𝚊𝚍𝚢 𝚛𝚎𝚌𝚘𝚛𝚍𝚎𝚍 𝚢𝚘𝚞𝚛 𝚒𝚗𝚒𝚝𝚒𝚊𝚕 𝚛𝚎𝚊𝚌𝚝𝚒𝚘𝚗. 𝙷𝚘𝚠𝚎𝚟𝚎𝚛, 𝚢𝚘𝚞 𝚌𝚊𝚗 𝚜𝚝𝚒𝚕𝚕 𝚛𝚎𝚊𝚌𝚝 𝚝𝚘 𝚊 𝚍𝚒𝚏𝚏𝚎𝚛𝚎𝚗𝚝 𝚙𝚘𝚜𝚝, 𝚜𝚊𝚖𝚎 𝚙𝚛𝚘𝚝𝚘𝚌𝚘𝚕 𝚠𝚎'𝚛𝚎 𝚞𝚜𝚒𝚗𝚐 𝚝𝚘 𝚏𝚘𝚕𝚕𝚘𝚠."
            	self.sendmessage(author_id, thread_id, thread_type, reply)
            if ("you from" in msg):
                reply = "I am from Philippines, currently living in cagayan de oro."
                self.sendmessage(author_id, thread_id, thread_type, reply)
            if msg.startswith(f"{prefix}echo"):
                echo_text = msg[len(prefix)+5:]
                reply = f"{echo_text}"
                self.sendmessage(author_id, thread_id, thread_type, reply)

        except Exception as e:
            print(f"Error: {e}")
            
follow_in_progress = False
reaction_in_progress = False

def rainbow_light_text_print(text, end='\n'):
    colors = [
        "\033[91m",  
        "\033[93m",  
        "\033[92m",  
        "\033[96m",  
        "\033[94m",  
        "\033[95m",  
    ]

    num_steps = len(colors)

    for i, char in enumerate(text):
        color_index = i % num_steps
        print(f"{colors[color_index]}{char}", end="")

    print("\033[0m", end=end)

def convert_cookie(session):
    return '; '.join([f"{cookie['name']}={cookie['value']}" for cookie in session])

if __name__ == '__main__':
    with open('configuration.json') as f:
    	configuration = json.load(f)
    try:
        form = {
            'adid': 'e3a395f9-84b6-44f6-a0ce-fe83e934fd4d',
            'email': str(configuration['CONFIG']['BOT_INFO']['EMAIL']),
            'password': str(configuration['CONFIG']['BOT_INFO']['PASSWORD']),
            'format': 'json',
            'device_id': '67f431b8-640b-4f73-a077-acc5d3125b21',
            'cpl': 'true',
            'family_device_id': '67f431b8-640b-4f73-a077-acc5d3125b21',
            'locale': 'en_US',
            'client_country_code': 'US',
            'credentials_type': 'device_based_login_password',
            'generate_session_cookies': '1',
            'generate_analytics_claim': '1',
            'generate_machine_id': '1',
            'currently_logged_in_userid': '0',
            'irisSeqID': 1,
            'try_num': '1',
            'enroll_misauth': 'false',
            'meta_inf_fbmeta': 'NO_FILE',
            'source': 'login',
            'machine_id': 'KBz5fEj0GAvVAhtufg3nMDYG',
            'meta_inf_fbmeta': '',
            'fb_api_req_friendly_name': 'authenticate',
            'fb_api_caller_class': 'com.facebook.account.login.protocol.Fb4aAuthHandler',
            'api_key': '882a8490361da98702bf97a021ddc14d',
            'access_token': '181425161904154|95a15d22a0e735b2983ecb9759dbaf91'
        }

        headers = {
            'content-type': 'application/x-www-form-urlencoded',
            'x-fb-friendly-name': form['fb_api_req_friendly_name'],
            'x-fb-http-engine': 'Liger',
            'user-agent': str(ua_generator.generate())
        }

        url = 'https://b-graph.facebook.com/auth/login'
        response = requests.post(url, data=form, headers=headers)
        response_data = response.json()
        #print(response_data)
        if "access_token" in response_data:
            access_token = response_data['access_token']
            cookie = convert_cookie(response_data['session_cookies'])
            key_value_pairs = [pair.strip() for pair in cookie.split(";")]
            session_cookies = {key: value for key, value in (pair.split("=") for pair in key_value_pairs)}
            rainbow_light_text_print("[ [ NAME ] ] FB BOOSTING CHATBOT")
            rainbow_light_text_print("[ [ VERSION ] ] Version: 1.0.2")
            time.sleep(0.5)
            rainbow_light_text_print("[ [ DESCRIPTION ] ] A Facebook Messenger Bot that send reaction and follow via page using fb accounts.")
            if str(configuration['CONFIG']['BOT_INFO']['PREFIX']) == "" or " " in configuration['CONFIG']['BOT_INFO']['PREFIX'] or len(configuration['CONFIG']['BOT_INFO']['PREFIX']) != 1:
            	sys.exit("\033[91m[ [ ERROR ] ] PLEASE CHECK THE PREFIX, PREFIX MUST HAVE VALUE AND DOESN'T HAVE SPACE AND ONLY ONE SYMBOL/LETTER. \033[0m")
            else:
            	try:
            		bot = MessBot(' ', ' ', session_cookies=session_cookies)
            		rainbow_light_text_print("[ [ CONNECTING ] ] {}".format(str(bot.isLoggedIn()).upper()))
            	except:
            		sys.exit("\033[91m[ [ ERROR ] ] FAILED TO CONNECT TO SERVER, TRY TO RERUN TO PROGRAM. \033[0m")
            	try:
            		bot.listen()
            	except:
            		bot.listen()
        else:
            rainbow_light_text_print("[ [ ERROR ] ] {}".format(str(response_data['error']['message'])))
    except requests.exceptions.ConnectionError:
    	print("\033[1m\033[91mPLEASE CHECK YOUR INTERNET CONNECTION AND TRY AGAIN.\033[0m")
