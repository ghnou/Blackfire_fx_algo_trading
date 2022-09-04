import pandas as pd
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from borb.pdf.pdf import PDF
from borb.pdf import Document

from borb.pdf.page.page import Page
from borb.pdf.canvas.geometry.rectangle import Rectangle

from borb.pdf.canvas.layout.table.table import TableCell
from borb.pdf.canvas.layout.table.fixed_column_width_table import FixedColumnWidthTable as Table

from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.canvas.layout.image.image import Image
from borb.pdf.canvas.layout.image.chart import Chart
from borb.pdf.canvas.layout.layout_element import Alignment

from borb.pdf.canvas.color.color import HexColor
from fx_trading.utils import constants as cst

from borb.pdf.canvas.layout.annotation.square_annotation import SquareAnnotation


def build_title(title, text_color=HexColor('ffffff'), background_color=HexColor('000000'),
                font="Times-Bold", font_size=12, padding_left=5, alignment=Alignment.LEFT):

    table = Table(number_of_rows=1, number_of_columns=1, background_color=background_color,
                  margin_top=Decimal(0), padding_bottom=Decimal(5))
    table.add(
        Paragraph(
            title, font_color=text_color, font=font, text_alignment=alignment,
            margin_bottom=Decimal(15), margin_top=Decimal(0), font_size=Decimal(font_size),
            padding_left=Decimal(padding_left)
        )
    ).no_borders()

    return table


def add_text(text, text_color, font, font_size):

    table = Table(number_of_rows=1, number_of_columns=1, background_color=HexColor("000000"),
                  margin_top=Decimal(0), padding_bottom=Decimal(5))
    table.add(
        Paragraph(
            text, font_color=text_color, font=font, text_alignment=Alignment.JUSTIFIED,
            margin_bottom=Decimal(15), margin_top=Decimal(0), font_size=Decimal(font_size),
            padding_left=Decimal(5)
        )
    ).no_borders()

    return table

def plot_line_chart(df, params, dimensions, fig_size=(10, 10), perc=True, decimal=0):

    df.index = df.index.astype('datetime64[ns]')
    fig, ax = plt.subplots(figsize=fig_size)
    my_range = df.index

    for i in params:
        if i in df:
            ax.plot(my_range, df[i], color=params[i], alpha=1, linewidth=2, label=i)
    if perc:
        ax.yaxis.set_majort_formatter(PercentFormatter(decimals=decimal))
    plt.tight_layout()

    return Chart(
        plt.gcf(),
        width=dimensions[0],
        height=dimensions[1],
        horizontal_alignment=Alignment.CENTERED,
        vertical_alignment=Alignment.TOP
    )

def add_fig(fig, background_color=HexColor('ffffff')):

    table = Table(
        number_of_rows=1,
        number_of_columns=1,
        background_color=background_color
    )

    return table.add(fig).no_borders()

class CustomPage(Page):

    def __init__(self, title, subtitle):
        super(CustomPage, self).__init__()
        self.__build_header(title, subtitle)

    def __build_header(self, title, subtitle):

        business_date = pd.to_datetime('today').strftime('%Y-%m-%d')

        r: Rectangle = Rectangle(
            Decimal(20),
            Decimal(848 - 20 - 40),
            Decimal(595 - 20 * 2),
            Decimal(40),
        )

        table = Table(
            number_of_rows=2, number_of_columns=3, column_widths=[Decimal(0.2), Decimal(0.6), Decimal(0.2)],
            background_color=HexColor("000000")
        )

        table.add(
            TableCell(
                Image(cst.LOGO_PATH, width=Decimal(50), height=Decimal(40), margin_bottom=Decimal(0),
                      padding_left=Decimal(5), margin_right=Decimal(0), margin_top=Decimal(0)),
                row_span=2,
                padding_top=Decimal(5)
            )
        ).add(
            Paragraph(title, font_color=HexColor(cst.PRIMARY_COLOR), font=cst.PRIMARY_FONT,
                      padding_bottom=Decimal(0), text_alignment=Alignment.CENTERED,
                      font_size=Decimal(cst.PRIMARY_FONT_SIZE))
        ).add(
            Paragraph('Analyse Stratégie', font_color=HexColor(cst.PRIMARY_COLOR), font=cst.PRIMARY_FONT,
                      padding_top=Decimal(10), text_alignment=Alignment.RIGHT, padding_right=Decimal(5),
                      font_size=Decimal(8))
        ).add(
            Paragraph(subtitle, font_color=HexColor(cst.SECONDARY_COLOR), font=cst.SECONDARY_FONT,
                      padding_bottom=Decimal(15), text_alignment=Alignment.CENTERED,
                      font_size=Decimal(cst.SECONDARY_FONT_SIZE))
        ).add(
            Paragraph(business_date, font_color=HexColor(cst.PRIMARY_COLOR), font=cst.SECONDARY_FONT,
                      padding_bottom=Decimal(7), text_alignment=Alignment.RIGHT, padding_right=Decimal(5),
                      font_size=Decimal(8))
        ).no_borders()

        table.layout(self, r)


class BuildTables:

    def __init__(self, pdf, fx_pair):

        self.pdf = pdf
        self.__fx_pair = fx_pair
        # self.stats_path = stats_path
        # self.strategy_name = strategy_name
        self.__x_min = cst.X_BEGIN
        self.__height = 0
        self.__y_min = cst.Y_BEGIN
        self.__x_max = cst.X_END

    def __reset_pos(self):
        self.__x_min = cst.X_BEGIN
        self.__height = 0
        self.__y_min = cst.Y_BEGIN
        self.__x_max = cst.X_END

    def __add_pos(self, x_min, height, x_max, padding=0):

        self.__x_min = x_min
        self.__y_min -= (self.__height + padding)
        self.__height = height
        self.__x_max = x_max

        r = Rectangle(Decimal(self.__x_min), Decimal(self.__y_min - self.__height),
                      Decimal(self.__x_max), Decimal(self.__height))

        return r

    @staticmethod
    def get_data(path):

        return pd.read_parquet(path)

    @staticmethod
    def build_raw_table(df, columns_widths):

        table = Table(
            number_of_rows=len(df) + df.columns.nlevels, number_of_columns=df.shape[1],
            border_color=HexColor("bbbbbb"), column_widths=columns_widths,
            margin_bottom=Decimal(0), margin_top=Decimal(0), margin_right=Decimal(0),
            margin_left=Decimal(0), border_width=Decimal(0)
        )

        if df.columns.nlevels > 1:
            for level in range(df.columns.nlevels):
                all_level = df.rename(columns={'': 'AA'}, level=-1).columns.get_level_values(level)
                unique = all_level.unique()
                if level == 0:
                    for row_value in range(len(unique)):
                        table.add(
                            TableCell(
                                Paragraph(
                                    unique[row_value], font_color=HexColor("000000"), font=cst.PRIMARY_FONT,
                                    font_size=Decimal(10),text_alignment=Alignment.CENTERED,
                                ),
                                border_top=True,
                                border_left=False,
                                border_right=False,
                                border_bottom=False,
                                col_span=all_level.tolist().count(unique[row_value])
                            )
                        )
                if level == df.columns.nlevels - 1:
                    for row_value in range(len(all_level)):
                        table.add(
                            TableCell(
                                Paragraph(
                                    all_level[row_value], font_color=HexColor("000000"), font='Times-roman',
                                    font_size=Decimal(cst.TEXT_TABLE_FONT_SIZE), text_alignment=Alignment.JUSTIFIED,
                                    padding_right=Decimal(5), vertical_alignment=Alignment.TOP,
                                    horizontal_alignment=Alignment.LEFT, padding_bottom=Decimal(5),
                                    padding_left=Decimal(5)
                                ),
                                border_right=False,
                                border_bottom=False,
                                border_top=False,
                                border_left=False
                            )
                        )

        to_write = df.fillna(0).values

        for j in range(len(to_write)):
            value = to_write[j]
            value = list(map(str, value))
            for i in range(len(value)):
                if (i == 0) & (j != len(to_write) - 1):
                    table.add(
                        TableCell(
                            Paragraph(
                                value[i], font_color=HexColor("000000"), font='Times-roman',
                                font_size=Decimal(cst.TEXT_TABLE_FONT_SIZE), text_alignment=Alignment.JUSTIFIED,
                                padding_right=Decimal(5), vertical_alignment=Alignment.TOP,
                                horizontal_alignment=Alignment.LEFT, padding_bottom=Decimal(5), padding_left=Decimal(5)
                            ),
                            border_right=False,
                            border_bottom=False,
                            border_top=False,
                            border_left=False
                        )
                    )
                elif (i == 0) & (j == len(to_write) - 1):
                    table.add(
                        TableCell(
                            Paragraph(
                                value[i], font_color=HexColor("000000"), font='Times-roman',
                                font_size=Decimal(cst.TEXT_TABLE_FONT_SIZE), text_alignment=Alignment.JUSTIFIED,
                                padding_right=Decimal(5), vertical_alignment=Alignment.TOP,
                                horizontal_alignment=Alignment.LEFT, padding_bottom=Decimal(5), padding_left=Decimal(5)
                            ),
                            border_right=False,
                            border_bottom=True,
                            border_top=False,
                            border_left=False
                        )
                    )
                elif j == 0:
                    table.add(
                        TableCell(
                            Paragraph(
                                value[i], font_color=HexColor("000000"), font='Times-roman',
                                font_size=Decimal(cst.TEXT_TABLE_FONT_SIZE), text_alignment=Alignment.JUSTIFIED,
                                padding_right=Decimal(5), vertical_alignment=Alignment.TOP,
                                horizontal_alignment=Alignment.LEFT, padding_bottom=Decimal(5), padding_left=Decimal(5)
                            ),
                            border_right=False,
                            border_bottom=False,
                            border_top=True,
                            border_left=False
                        )
                    )
                elif j == len(to_write) - 1:
                    table.add(
                        TableCell(
                            Paragraph(
                                value[i], font_color=HexColor("000000"), font='Times-roman',
                                font_size=Decimal(cst.TEXT_TABLE_FONT_SIZE), text_alignment=Alignment.JUSTIFIED,
                                padding_right=Decimal(5), vertical_alignment=Alignment.TOP,
                                horizontal_alignment=Alignment.LEFT, padding_bottom=Decimal(5), padding_left=Decimal(5)
                            ),
                            border_right=False,
                            border_bottom=True,
                            border_top=False,
                            border_left=False
                        )
                    )
                else:
                    table.add(
                        TableCell(
                            Paragraph(
                                value[i], font_color=HexColor("000000"), font='Times-roman',
                                font_size=Decimal(cst.TEXT_TABLE_FONT_SIZE), text_alignment=Alignment.JUSTIFIED,
                                padding_right=Decimal(5), vertical_alignment=Alignment.TOP,
                                horizontal_alignment=Alignment.LEFT, padding_bottom=Decimal(5), padding_left=Decimal(5)
                            ),
                            border_right=False,
                            border_bottom=False,
                            border_top=False,
                            border_left=False
                        )
                    )

        return table

    def build_page(self, df, subtitle, title, font=cst.HEADER_FONT, font_size=cst.HEADER_FONT_SIZE):

        page = CustomPage(title=cst.HEAD_TITLE, subtitle=subtitle)
        self.__reset_pos()
        title = build_title(title, text_color=HexColor("000000"), background_color=HexColor('ffffff'),
                            font=font, font_size=font_size, padding_left=5,
                            alignment=Alignment.CENTERED)
        title.layout(page, self.__add_pos(cst.X_BEGIN + 20, 40, cst.X_END - 40, 10))

        n = df.shape[1]
        columns_widths = n * [Decimal(1 / n)]
        table = self.build_raw_table(df, columns_widths=columns_widths)
        r = self.__add_pos(cst.X_BEGIN + 20, 50, cst.X_END - 40, 0)
        table.layout(page, r)

        return page

    def intro_page(self):

        page = Page()
        self.__reset_pos()

        table = Table(
            number_of_rows=1, number_of_columns=1,
            background_color=HexColor("000000")
        )

        table.add(
            TableCell(
                Image(cst.LOGO_PATH, width=Decimal(500), height=Decimal(500), margin_bottom=Decimal(0),
                      padding_left=Decimal(5), margin_right=Decimal(0), margin_top=Decimal(0),
                      vertical_alignment=Alignment.TOP, horizontal_alignment=Alignment.CENTERED,
                      ),
                padding_top=Decimal(5)
            )
        ).layout(page, self.__add_pos(cst.X_BEGIN + 20, 500, cst.X_END - 40, 10))

        title = f'Descriptive statistics and results for the FX pair: {cst.ALIAS[self.__fx_pair]}'
        title = build_title(title, text_color=HexColor("000000"), background_color=HexColor('ffffff'),
                            font=cst.HEADER_FONT, font_size=24, padding_left=5,
                            alignment=Alignment.CENTERED)
        title.layout(page, self.__add_pos(cst.X_BEGIN + 20, 40, cst.X_END - 40, 10))

        self.pdf.add_page(page)

    def build_hist_chart(self):

        page = CustomPage(title=cst.HEAD_TITLE, subtitle='Historical Close Price')
        self.__reset_pos()
        title = f"""
                This chart present the historical close price of the FX pair: {cst.ALIAS[self.__fx_pair]}."""
        title = build_title(title, text_color=HexColor("000000"), background_color=HexColor('ffffff'),
                            font=cst.HEADER_FONT, font_size=cst.SECONDARY_FONT_SIZE, padding_left=5,
                            alignment=Alignment.CENTERED)
        title.layout(page, self.__add_pos(cst.X_BEGIN + 20, 40, cst.X_END - 40, 10))

        path = cst.FX_STATS_DESCRIPTIVE.format(cst.ASSET[self.__fx_pair], self.__fx_pair)
        df = self.get_data(path + 'plot.parquet')
        fig = add_fig(plot_line_chart(df, cst.COLORS_RGB, [Decimal(500), Decimal(260)],
                                      fig_size=(13, 7.5), perc=False))

        fig.layout(page, self.__add_pos(cst.X_BEGIN + 20, 500, cst.X_END - 40, 10))

        self.pdf.add_page(page)

    def descriptive_page(self,):

        path = cst.FX_STATS_DESCRIPTIVE.format(cst.ASSET[self.__fx_pair], self.__fx_pair)
        df = self.get_data(path + 'stats.parquet')
        subtitle = "FX statistic descriptive"
        title = f"""
        This table presents the average candle size, bid-ask spread of the pair {cst.ALIAS[self.__fx_pair]} for 
        the London, New York and Asian session."""

        self.pdf.add_page(self.build_page(df, subtitle, title, font_size=cst.SECONDARY_FONT_SIZE))

        pass

    def run_candle_sticks_entry_stats(self):

        path = cst.FX_STATS_CANDLESTICK.format(cst.ASSET[self.__fx_pair] ,self.__fx_pair)
        df = self.get_data(path + 'potential_entry.parquet')
        df.sort_values([('Candle_name', ), ('Candle_type', )], inplace=True)

        for group, stats in df.groupby([cst.FX_PAIR, 'Candle_name', 'Candle_type']):
            print(group)
            stats.drop(columns=[('FX_PAIR',), ('Candle_type',), ('Candle_name',), ('Stats',)], inplace=True)
            stats = stats.sort_values(('DATE', )).fillna('-')

            title = f"""Potential Entry for : FX Pair: {cst.ALIAS[group[0]]}, Candle Name: {group[1]}, 
                    Candle Type: {group[2]}"""
            subtitle = 'Candlesticks Analysis'

            self.pdf.add_page(self.build_page(stats, subtitle, title))

    def run_candle_max_gain(self):

        path = cst.FX_STATS_CANDLESTICK.format(cst.ASSET[self.__fx_pair] ,self.__fx_pair)
        df = self.get_data(path + 'max_g.parquet')
        df.sort_values([('Candle_name',), ('Candle_type',)], inplace=True)

        for group, stats in df.groupby([cst.FX_PAIR, 'Candle_name', 'Candle_type']):
            print(group)
            stats.drop(columns=[('FX_PAIR',), ('Candle_type',), ('Candle_name',), ('Stats',)], inplace=True)
            stats = stats.sort_values(('DATE',)).fillna('-')

            title = f"""Max Gain for : FX Pair: {cst.ALIAS[group[0]]}, Candle Name: {group[1]}, 
                    Candle Type: {group[2]}"""
            subtitle = 'Candlesticks Analysis'

            self.pdf.add_page(self.build_page(stats, subtitle, title))

    def run(self):

        self.intro_page()
        self.build_hist_chart()
        self.descriptive_page()
        # self.run_candle_max_gain()
        # self.run_candle_sticks_entry_stats()


class BuildStrategyReport:

    def __init__(self, fx_pair):

        self.__fx_pair = fx_pair

    def run(self):

        pdf = Document()
        BuildTables(pdf, self.__fx_pair).run()
        # for fx_pair in cst.FX_PIP:
        #     BuildTables(pdf, fx_pair).run()
        #     print(f'Done for {fx_pair}')
        path = cst.FX_STATS_ROOT.format(cst.ASSET[self.__fx_pair], self.__fx_pair)
        with open(path, 'wb') as pdf_file:
            PDF.dumps(pdf_file, pdf)


if __name__ == '__main__':
    BuildStrategyReport('eurusd').run()