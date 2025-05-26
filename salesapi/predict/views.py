import joblib
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response

# Load model and data

model = joblib.load('model/xgb_model.pkl')

monthly_df = pd.read_csv("data/monthly_with_features.csv")

class PredictView(APIView):
    def post(self, request):
        shop_id = int(request.data.get('shop_id'))
        item_id = int(request.data.get('item_id'))
        date_block_num = int(request.data.get('date_block_num'))

        record = monthly_df[
            (monthly_df['shop_id'] == shop_id) &
            (monthly_df['item_id'] == item_id)
        ].sort_values('date_block_num').tail(1)

        if record.empty:
            features = pd.DataFrame([{
                'shop_id': shop_id,
                'item_id': item_id,
                'date_block_num': date_block_num,
                'item_price': 0,
                'item_cnt_month_lag_1': 0,
                'item_cnt_month_lag_2': 0,
                'item_cnt_month_lag_3': 0,
                'item_price_lag_1': 0,
                'item_cnt_month_rolling': 0
            }])
        else:
            r = record.iloc[0]
            features = pd.DataFrame([{
                'shop_id': shop_id,
                'item_id': item_id,
                'date_block_num': date_block_num,
                'item_price': r['item_price_lag_1'],
                'item_cnt_month_lag_1': r['item_cnt_month_lag_1'],
                'item_cnt_month_lag_2': r['item_cnt_month_lag_2'],
                'item_cnt_month_lag_3': r['item_cnt_month_lag_3'],
                'item_price_lag_1': r['item_price_lag_1'],
                'item_cnt_month_rolling': r['item_cnt_month_rolling']
            }])

        prediction = model.predict(features)[0]
        prediction = max(0, min(20, prediction))

        return Response({"predicted_item_cnt_month": round(float(prediction), 2)})
