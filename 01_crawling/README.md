# 📥 Crawling Module

## Files
- `crawlcode.ipynb`: Notebook để crawl dữ liệu từ VnExpress
- `updata.py`: Script upload data lên MongoDB

## Usage
1. Chạy `crawlcode.ipynb` để crawl data
2. Data được lưu vào `data/raw/`
3. Chạy `python updata.py` để upload lên MongoDB

## Output
- vnexpress_bongda_part1.json
- vnexpress_bongda_part2.json  
- vnexpress_bongda_part3.json
- vnexpress_bongda_part4.json

Total: ~1830+ articles
