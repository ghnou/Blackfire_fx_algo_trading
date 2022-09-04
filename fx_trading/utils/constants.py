from pathlib import Path
import warnings
import pandas as pd
from matplotlib import cm, rc, gridspec
import matplotlib as mpl
import matplotlib.dates as mdates
# from matplotlib.ticker import Funcformatter

rc('font', **{'sans-serif': ['Arial'], 'family': 'monospace', 'weight': 'bold'})
# rc('axes', line)
mpl.rcParams['font.weight'] = 'heavy'
mpl.rcParams['axes.linewidth'] = 3
mpl.rcParams['patch.linewidth'] = 3
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['ytick.major.pad'] = 8
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['axes.titlesize'] = 10




pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")

RAW_DATA_PATH = str(Path(__file__).parent.parent.parent.parent.parent) + 'BLKFR_DATA/FX'
FX_15MIN_DATA_PATH = RAW_DATA_PATH + '/fx_15min'
FX_1H_DATA_PATH = RAW_DATA_PATH + 'fx_data/fx_1h'
FX_4H_DATA_PATH = RAW_DATA_PATH + 'fx_data/fx_4h'
FX_1D_DATA_PATH = RAW_DATA_PATH + 'fx_data/fx_1D'
FX_TICK_CHUNK_DATA_PATH = RAW_DATA_PATH + 'fx_data/fx_tick_per_chunk'

# Statistics
RAW_STATS_PATH = str(Path(__file__).parent.parent.parent) + '/'
FX_STATS_CANDLESTICK = RAW_STATS_PATH + 'fx_stats/{}/{}/candlestick_pattern/'
FX_STATS_DESCRIPTIVE = RAW_STATS_PATH + 'fx_stats/{}/{}/descriptive_stats/'
FX_STATS_ROOT = RAW_STATS_PATH + 'fx_stats/{}/{}/report.pdf'


DATE = 'DATE'
OPEN = 'OpenLAST'
HIGH = 'HighLAST'
LOW = 'LowLAST'
CLOSE = 'CloseLAST'
PAIR = 'FX_PAIR'
CANDLESIZE = 'CandleSIZE'
CANDLECOLOR = 'Candle'
REAL_VOL = 'Realized Vol'
MOVING_AVERAGE_100D = 'MA100'

FX = 'FOREX'
COMMODITY = 'COMMODITY'
INDEX = 'INDEX'

FX_PIP = {'eurusd': 10_000, 'eurchf': 10_000, 'eurgbp': 10_000, 'eurjpy': 100, 'euraud': 10_000, 'usdcad': 10_000,
          'usdchf': 10_000, 'usdjpy': 100, 'usdmxn': 10_000, 'gbpchf': 10_000, 'gbpjpy': 100, 'gbpusd': 10_000,
          'audjpy': 100, 'audusd': 10_000, 'chfjpy': 100, 'nzdjpy': 100, 'nzdusd': 10_000, 'xauusd': 100,
          'eurcad': 10_000, 'audcad': 10_000, 'cadjpy': 100, 'eurnzd': 10_000, 'grxeur': 1, 'nzdcad': 10_000,
          'usdhkd': 10_000, 'usdnok': 10_000, 'usdtry': 10_000, 'xauaud': 100, 'audchf': 10_000,
          'auxaud': 1, 'eurhuf': 100, 'eurpln': 10_000, 'frxeur': 1, 'hkxhkd': 1, 'nzdchf': 10_000,
          'spxusd': 1, 'usdhuf': 10_000, 'usdpln': 10_000, 'usdzar': 10_000, 'xauchf': 100, 'zarjpy': 100,
          'bcousd': 100, 'etxeur': 100, 'eurczk': 10_000, 'eursek': 10_000, 'gbpaud': 10_000, 'gbpnzd': 10_000,
          'jpxjpy': 1, 'udxusd': 10_000, 'usdczk': 10_000, 'usdsek': 10_000, 'wtiusd': 1, 'xaueur': 100,
          'audnzd': 10_000, 'cadchf': 10_000, 'eurdkk': 10_000, 'eurnok': 10_000, 'eurtry': 10_000, 'gbpcad': 10_000,
          'nsxusd': 1, 'ukxgbp': 1, 'usddkk': 10_000, 'usdsgd': 10_000, 'xagusd': 10_000}

ALIAS = {'eurusd': 'EUR/USD', 'eurchf': 'EUR/CHF', 'eurgbp': 'EUR/GBP', 'eurjpy': 'EUR/JPY', 'euraud': 'EUR/AUD',
         'usdcad': 'USD/CAD', 'usdchf': 'USD/CHF', 'usdjpy': 'USD/JPY', 'usdmxn': 'USD/MXN', 'gbpchf': 'GBP/CHF',
         'gbpjpy': 'GBP/JPY', 'gbpusd': 'GBP/USD', 'audjpy': 'AUD/JPY', 'audusd': 'AUD/USD', 'chfjpy': 'CHF/JPY',
         'nzdjpy': 'NZD/JPY', 'nzdusd': 'NZD/USD', 'xauusd': 'XAU/USD', 'eurcad': 'EUR/CAD', 'audcad': 'AUD/CAD',
         'cadjpy': 'CAD/JPY', 'eurnzd': 'EUR/NZD', 'grxeur': 'GRX/EUR', 'nzdcad': 'NZD/CAD', 'sgdjpy': 'SGD/JPY',
         'usdhkd': 'USD/HKD', 'usdnok': 'USD/NOK', 'usdtry': 'USD/TRY', 'xauaud': 'XAU/AUD', 'audchf': 'AUD/CHF',
         'auxaud': 'AUX/AUD', 'eurhuf': 'EUR/HUF', 'eurpln': 'EUR/PLN', 'frxeur': 'FRX/EUR', 'hkxhkd': 'HKX/HKD',
         'nzdchf': 'NZD/CHF', 'spxusd': 'SPX/USD', 'usdhuf': 'USD/HUF', 'usdpln': 'USD/PLN', 'usdzar': 'USD/ZAR',
         'xauchf': 'XAU/CHF', 'zarjpy': 'ZAR/JPY', 'bcousd': 'BCO/USD', 'etxeur': 'ETX/EUR', 'eurczk': 'EUR/CZK',
         'eursek': 'EUR/SEK', 'gbpaud': 'GBP/AUD', 'gbpnzd': 'GBP/NZD', 'jpxjpy': 'JPX/JPY', 'udxusd': 'UDX/USD',
         'usdczk': 'USD/CZK', 'usdsek': 'USD/SEK', 'wtiusd': 'WTI/USD', 'xaueur': 'XAU/EUR', 'audnzd': 'AUD/NZD',
         'cadchf': 'CAD/CHF', 'eurdkk': 'EUR/DKK', 'eurnok': 'EUR/NOK', 'eurtry': 'EUR/TRY', 'gbpcad': 'GBP/CAD',
         'nsxusd': 'NSX/USD', 'ukxgbp': 'UKX/GBP', 'usddkk': 'USD/DKK', 'usdsgd': 'USD/SGD', 'xagusd': 'XAG/USD',
         'xaugbp': 'XAU/GBP'}

ASSET = {'eurusd': FX, 'eurchf': FX, 'eurgbp': FX, 'eurjpy': FX, 'euraud': FX, 'usdcad': FX, 'usdchf': FX,
         'usdjpy': FX, 'usdmxn': FX, 'gbpchf': FX, 'gbpjpy': FX, 'gbpusd': FX, 'audjpy': FX, 'audusd': FX,
         'chfjpy': FX, 'nzdjpy': FX, 'nzdusd': FX, 'xauusd': COMMODITY, 'eurcad': FX, 'audcad': FX,
         'cadjpy': FX, 'eurnzd': FX, 'grxeur': INDEX, 'nzdcad': FX, 'sgdjpy': FX, 'usdhkd': FX,
         'usdnok': FX, 'usdtry': FX, 'xauaud': COMMODITY, 'audchf': FX, 'auxaud': INDEX, 'eurhuf': FX,
         'eurpln': FX, 'frxeur': INDEX, 'hkxhkd': INDEX, 'nzdchf': FX, 'spxusd': INDEX, 'usdhuf': FX,
         'usdpln': FX, 'usdzar': FX, 'xauchf': COMMODITY, 'zarjpy': FX, 'bcousd': COMMODITY, 'etxeur': FX,
         'eurczk': FX, 'eursek': FX, 'gbpaud': FX, 'gbpnzd': FX, 'jpxjpy': INDEX, 'udxusd': FX,
         'usdczk': FX, 'usdsek': FX, 'wtiusd': COMMODITY, 'xaueur': COMMODITY, 'audnzd': FX,
         'cadchf': FX, 'eurdkk': FX, 'eurnok': FX, 'eurtry': FX, 'gbpcad': FX, 'nsxusd': INDEX,
         'ukxgbp': INDEX, 'usddkk': FX, 'usdsgd': FX, 'xagusd': COMMODITY, 'xaugbp': COMMODITY}

#'sgdjpy': 100
# 'xaugbp': 100

BULLISH = 'BULLISH'
BEARISH = 'BEARISH'
FX_PAIR = 'FX_PAIR'

LOGO_PATH = Path(RAW_STATS_PATH + 'fx_trading/utils/black.png')
PRIMARY_COLOR = "ffffff"
PRIMARY_FONT = "Times-Bold"
PRIMARY_FONT_SIZE = 18

SECONDARY_COLOR = "F79525"
SECONDARY_FONT = "Times-Italic"
SECONDARY_FONT_SIZE = 14

TEXT_TABLE_FONT_SIZE = 8
HEAD_TITLE = 'BLACKFIRE CAPITAL INC'
X_BEGIN = 20
X_END = 555
Y_BEGIN = 778

HEADER_FONT = "Times-Bold"
HEADER_FONT_SIZE = 12

COLORS_RGB = {CLOSE: (247/255, 149/255, 37/255)}
