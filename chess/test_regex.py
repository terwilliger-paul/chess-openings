import re
import chess
from chess import svg
import datetime
import pandas as pd

today = datetime.datetime.today()

text = '[%weight 1.0] [%streak 0.0] [%datetime {}]'.format(today)

m = re.search('\[%datetime (.+?)]', text)
if m:
    found = m.group(1)
    print found
    print pd.to_datetime(found).to_pydatetime()
    print pd.to_datetime(found).to_pydatetime() == today

print(re.sub('\[%streak (.+?)]', '', text))

#from IPython.display import SVG, display
#def show_svg():
#    return display(SVG(svg.piece(chess.Piece.from_symbol("R"))))
#show_svg()
#print('hi')

print text
