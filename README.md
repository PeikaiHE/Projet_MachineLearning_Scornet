# Projet_MachineLearning 
**E.Scornet**

- Kaggle 链接: https://www.kaggle.com/competitions/optiver-trading-at-the-close
- Overleaf 链接：https://www.overleaf.com/8326179925tkwfmcgpythz#a0551b

## 任务分配？
- 任务1: 数据预处理，数据可视化，
- 任务2: 特征工程
- 任务3: 模型训练，模型融合

## Columns description
`Unnamed`: 0: Seems to be an index or identifier. \
`stock_id`': A unique identifier for the stock. \
`date_id`: A unique identifier for the date.\
`seconds_in_bucket`: The number of seconds elapsed since the beginning of the day's closing auction. \
`imbalance_size`: The amount unmatched at the current reference price (in USD).\
`imbalance_buy_sell_flag`: An indicator reflecting the direction of auction imbalance.\
`reference_price`: The reference price as defined.\
`matched_size`: The amount that can be matched at the current reference price (in USD).\
`far_price`: The far price as defined.\
`near_price`: The near price as defined.\
`bid_price`: Price of the most competitive buy level in the non-auction book.\
`bid_size`: The dollar notional amount on the most competitive buy level in the non-auction book.\
`ask_price`: Price of the most competitive sell level in the non-auction book.\
`ask_size`: The dollar notional amount on the most competitive sell level in the non-auction book.\
`wap`: The weighted average price in the non-auction book.\
`target`: The target value to be predicted.\
`time_id`: Appears to be another time-related identifier.\
`row_id`: A combined identifier.\