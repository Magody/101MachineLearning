#!/bin/bash
conda activate ridetel_chatbot
export LC_ALL=en_US.UTF-8
USERNAME="$1"
HOST="$2"
MONGO_DB_PORT="$3"
DB_NAME="$4"
MONGODB_USER="$5"
MONGODB_PASSWORD="$6"
APP_PORT="$7"
ENDPOINT_CHATBOT_BACKEND="$8"
WEB_SOCKET_CHATBOT_BACKEND="$9"
FORCE="${10}"
MODE="${11}"
DIRECTORY_FRONTEND="/var/www/html/production/frontend_user_${USERNAME}"
DIRECTORY_BACKEND="/var/production/backend_user_${USERNAME}"

if [ "${FORCE}" = "true" ]; then
  echo "WARNING: YOU ARE FORCING, FILES WILL BE DELETED. PROGRAM WILL WAIT 5 SECONDS BEFORE CONTINUE";
  sleep 5
fi
if [[ -d "${DIRECTORY_FRONTEND}" && "${FORCE}" == "false" ]]; then
  echo "ERROR $DIRECTORY_FRONTEND ALREADY EXISTS" 1>&2
  return 64
fi
if [[ -d "${DIRECTORY_BACKEND}" && "${FORCE}" == "false" ]]; then
  echo "ERROR $DIRECTORY_BACKEND ALREADY EXISTS" 1>&2
  return 64
fi
rm -rf "${DIRECTORY_FRONTEND}"
rm -rf "${DIRECTORY_BACKEND}"
git clone https://github.com/rideteldev1/RidetelChatbotLaravel.git "${DIRECTORY_FRONTEND}"
git clone https://github.com/rideteldev1/UserBrains.git "${DIRECTORY_BACKEND}"
echo "${DIRECTORY_FRONTEND}, ${DIRECTORY_BACKEND} cloned"
echo "Generating files for backend..."
echo "{ \"USERNAME\": \"${USERNAME}\", \"HOST\": \"${HOST}\", \"MONGO_DB_PORT\": ${MONGO_DB_PORT}, \"DB_NAME\": \"${DB_NAME}\", \"MONGODB_USER\": \"${MONGODB_USER}\", \"MONGODB_PASSWORD\": \"${MONGODB_PASSWORD}\", \"APP_PORT\": ${APP_PORT}}" > "${DIRECTORY_BACKEND}/.env.json"
echo ".env.json generated in ${DIRECTORY_BACKEND}"
cp "${DIRECTORY_BACKEND}/models/custom_bussiness.example.py"  "${DIRECTORY_BACKEND}/models/custom_bussiness.py"
echo "custom_bussiness.py generated in ${DIRECTORY_BACKEND}/models, make sure you customize its contents"
echo "Generating .env for frontend, and setup of composer key:generate"
cp "${DIRECTORY_FRONTEND}/.env.base"  "${DIRECTORY_FRONTEND}/.env"
printf "ENDPOINT_CHATBOT_BACKEND=${ENDPOINT_CHATBOT_BACKEND}\nWEB_SOCKET_CHATBOT_BACKEND=${WEB_SOCKET_CHATBOT_BACKEND}\nDB_HOST=${HOST}\nDB_PORT=${MONGO_DB_PORT}\nDB_DATABASE=${DB_NAME}\nDB_USERNAME=${MONGODB_USER}\nDB_PASSWORD=${MONGODB_PASSWORD}\n" >> "${DIRECTORY_FRONTEND}/.env"
echo "Generated .env in ${DIRECTORY_FRONTEND}"
cd /var/www/html/production/frontend_user_magody
yes | composer install
php artisan key:generate
chmod -R 777 storage/
cd "/var/${MODE}/ApiUserBrains"
echo "Process completed"
