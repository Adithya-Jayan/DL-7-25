| Date       | News                                                        | Open   | High   | Low    | Close  | Volume   | Label    |
|------------|-------------------------------------------------------------|--------|--------|--------|--------|----------|----------|
| YYYY-MM-DD | Content of news articles affecting Gold price                | 1234.5 | 1250.7 | 1222.3 | 1240.1 | 100000   | 1 / 0 / -1 |

**Column Description:**
- **Date**: The date the news was released
- **News**: The content of news articles that could potentially affect the Gold price
- **Open**: The Gold price (in Rs) at the beginning of the day
- **High**: The highest Gold price (in Rs) reached during the day
- **Low**: The lowest Gold price (in Rs) reached during the day
- **Close**: The adjusted Gold price (in Rs) at the end of the day
- **Volume**: Total volume traded during the day (Optional)
- **Label**: The sentiment polarity of the news content  
  - 1: positive  
  - 0: neutral  
  - -1: negative

**Sample Data Row:**
| 2024-06-01 | "Gold prices likely to rise amid global uncertainty." | 50000 | 50500 | 49500 | 50200 | 1500 | 1 |

Data Preprocessing
===================

1) During Preprocessing of data, find the mean, median and maximum length of text (sequence length). This would help to tune hyper parameter on sequence lenght.

