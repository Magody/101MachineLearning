# CODE: https://www.kaggle.com/ashbellett/data-engineering/notebook
from pathlib import Path        # file paths
from typing import Union        # type hints
import numpy as np              # linear algebra
import pandas as pd             # dataframes
import matplotlib.pyplot as plt # visualisations
import seaborn as sns           # visualisations
from scipy import stats         # statistics

METRICS_SCHEMA = {
    "Video": {
        "name": "video_id",
        "data_type": str
    },
    "Video title": {
        "name": "title",
        "data_type": str
    },
    "Video pub­lish time": {
        "name": "date",
        "data_type": "datetime"
    },
    "Com­ments ad­ded": {
        "name": "comment_count",
        "data_type": int
    },
    "Shares": {
        "name": "share_count",
        "data_type": int
    },
    "Likes": {
        "name": "like_count",
        "data_type": int
    },
    "Dis­likes": {
        "name": "dislike_count",
        "data_type": int
    },
    "Sub­scribers gained": {
        "name": "subscribers_gained",
        "data_type": int
    },
    "Sub­scribers lost": {
        "name": "subscribers_lost",
        "data_type": int
    },
    "Sub­scribers": {
        "name": "net_subscribers",
        "data_type": int
    },
    "Views": {
        "name": "view_count",
        "data_type": int
    },
    "Av­er­age per­cent­age viewed (%)": {
        "name": "average_view_ratio",
        "data_type": float
    },
    "Av­er­age view dur­a­tion": {
        "name": "average_watch_time",
        "data_type": "timedelta"
    },
    "Watch time (hours)": {
        "name": "total_watch_time",
        "data_type": float
    },
    "Im­pres­sions": {
        "name": "impression_count",
        "data_type": int
    },
    "Im­pres­sions click-through rate (%)": {
        "name": "click_through_rate",
        "data_type": float
    },
    "RPM (USD)": {
        "name": "rpm",
        "data_type": float
    },
    "CPM (USD)": {
        "name": "cpm",
        "data_type": float
    },
    "Your es­tim­ated rev­en­ue (USD)": {
        "name": "estimated_revenue",
        "data_type": float
    }
}

COMMENTS_SCHEMA = {
    "Comment_ID": {
        "name": "comment_id",
        "data_type": str
    },
    "Date": {
        "name": "date",
        "data_type": "datetime"
    },
    "VidId": {
        "name": "video_id",
        "data_type": str
    },
    "user_ID": {
        "name": "user_id",
        "data_type": str
    },
    "Comments": {
        "name": "comment_text",
        "data_type": str
    },
    "Reply_Count": {
        "name": "reply_count",
        "data_type": int
    },
    "Like_Count": {
        "name": "like_count",
        "data_type": int
    }
}

PERFORMANCE_SCHEMA_COUNTRY = {
    "Date": {
        "name": "date",
        "data_type": "datetime"
    },
    "Country Code": {
        "name": "country",
        "data_type": str
    },
    "Is Subscribed":{
        "name": "is_subscribed",
        "data_type": bool
    },
    "Video Title": {
        "name": "title",
        "data_type": str
    },
    "External Video ID": {
        "name": "video_id",
        "data_type": str
    },
    "Video Length": {
        "name": "length",
        "data_type": int
    },
    "Thumbnail link": {
        "name": "thumbnail",
        "data_type": str
    },
    "Views": {
        "name": "view_count",
        "data_type": int
    },
    "Video Likes Added": {
        "name": "likes_added",
        "data_type": int
    },
    "Video Dislikes Added": {
        "name": "dislikes_added",
        "data_type": int
    },
    "Video Likes Removed": {
        "name": "likes_removed",
        "data_type": int
    },
    "User Subscriptions Added": {
        "name": "subscribers_gained",
        "data_type": int
    },
    "User Subscriptions Removed": {
        "name": "subscribers_lost",
        "data_type": int
    },
    "Average View Percentage": {
        "name": "average_view_ratio",
        "data_type": float
    },
    "Average Watch Time": {
        "name": "average_watch_time",
        "data_type": float
    },
    "User Comments Added": {
        "name": "comments_added",
        "data_type": int
    }
}



PERFORMANCE_SCHEMA = {
    "Date": {
        "name": "date",
        "data_type": "datetime"
    },
    "Video Title": {
        "name": "title",
        "data_type": str
    },
    "External Video ID": {
        "name": "video_id",
        "data_type": str
    },
    "Video Length": {
        "name": "length",
        "data_type": int
    },
    "Thumbnail link": {
        "name": "thumbnail",
        "data_type": str
    },
    "Views": {
        "name": "view_count",
        "data_type": int
    },
    "Video Likes Added": {
        "name": "likes_added",
        "data_type": int
    },
    "Video Dislikes Added": {
        "name": "dislikes_added",
        "data_type": int
    },
    "Video Likes Removed": {
        "name": "likes_removed",
        "data_type": int
    },
    "User Subscriptions Added": {
        "name": "subscribers_gained",
        "data_type": int
    },
    "User Subscriptions Removed": {
        "name": "subscribers_lost",
        "data_type": int
    },
    "Average View Percentage": {
        "name": "average_view_ratio",
        "data_type": float
    },
    "Average Watch Time": {
        "name": "average_watch_time",
        "data_type": float
    },
    "User Comments Added": {
        "name": "comments_added",
        "data_type": int
    }
}

def typecast_column(column: pd.Series, data_type: Union[type, str]):
    if data_type == 'datetime':
        result = pd.to_datetime(column)
    elif data_type == 'timedelta':
        result = column.apply(lambda row: np.int16(pd.Timedelta(row).seconds))
    elif data_type == int:
        result = column.astype(np.int32)
    elif data_type == float:
        result = column.astype(np.float16)
    else:
        result = column.astype(data_type)
    return result





metrics_column_map = {
    column: METRICS_SCHEMA[column]["name"]
    for column
    in METRICS_SCHEMA.keys()
}

comments_column_map = {
    column: COMMENTS_SCHEMA[column]["name"]
    for column
    in COMMENTS_SCHEMA.keys()
}


performance_country_column_map = {
    column: PERFORMANCE_SCHEMA_COUNTRY[column]["name"]
    for column
    in PERFORMANCE_SCHEMA_COUNTRY.keys()
}

performance_column_map = {
    column: PERFORMANCE_SCHEMA[column]["name"]
    for column
    in PERFORMANCE_SCHEMA.keys()
}