Python version 3.7

$ docker run -p 27050:27017 -v /home/magody/programming/python/data_science/projects/business/nlp/chatbot/intents/RetrievalChatbot/mongo_volume:/data/db --name nlp_brain_chatbot -it mongo

$ docker start nlp_brain_chatbot
$ docker stop nlp_brain_chatbot

$ docker exec -it nlp_brain_chatbot bash
-- $ mongo
-- $ use db_chatbot
-- $ db.createUser({user: "developer", pwd: "developer123", roles: [ { role: "userAdminAnyDatabase", db: "admin" }]})
-- $ db.createUser({user: "developer", pwd: "developer123", roles: [ { role: "userAdminAnyDatabase", db: "db_chatbot" }]})


run python nltk_downloads.py to download corpus


TODO: Check monitor
