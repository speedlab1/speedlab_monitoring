'''
 # @ Create Time: 2025-01-15 11:41:41.123992
'''

import pathlib
from dash import Dash, dcc, html, Input, Output
import dash_table
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import os
import threading
import sys
import logging

app = Dash(__name__, title="monitoring")

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

def load_data(data_file: str) -> pd.DataFrame:
    '''
    Load data from /data directory
    '''
    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath("data").resolve()
    return pd.read_csv(DATA_PATH.joinpath(data_file))
# print('ok')
d_cry=load_data('d_cry.csv')
d_cry = d_cry.set_index('Unnamed: 0').to_dict()
win_pct_cry=load_data('win_pct_cry.csv')
win_pct_cry.set_index('Unnamed: 0',inplace=True)
frames_all_cry=load_data('frames_all_cry.csv')
frames_all_cry.set_index('Unnamed: 0',inplace=True)
frames_all_cry.index = pd.to_datetime(frames_all_cry.index)
summary_table_cry=load_data('summary_table_cry.csv')
summary_table_cry.set_index('Unnamed: 0',inplace=True)
change_date_cry=load_data('change_date_cry.csv')
change_date_cry = change_date_cry.set_index('Unnamed: 0').to_dict()['0']
old_eq_cry=load_data('old_eq_cry.csv')
old_eq_cry.set_index('Unnamed: 0',inplace=True)
old_eq_cry.index = pd.to_datetime(old_eq_cry.index)
current_cry=load_data('current_cry.csv')
current_cry.set_index('Unnamed: 0',inplace=True)
current_cry_w=load_data('current_cry_w.csv')
current_cry_w.set_index('Unnamed: 0',inplace=True)



d_sto=load_data('d_sto.csv')
d_sto = d_sto.set_index('Unnamed: 0').to_dict()
win_pct_sto=load_data('win_pct_sto.csv')
win_pct_sto.set_index('Unnamed: 0',inplace=True)
frames_all_sto=load_data('frames_all_sto.csv')
frames_all_sto.set_index('Unnamed: 0',inplace=True)
frames_all_sto.index = pd.to_datetime(frames_all_sto.index)
summary_table_sto=load_data('summary_table_sto.csv')
summary_table_sto.set_index('Unnamed: 0',inplace=True)
change_date_sto=load_data('change_date_sto.csv')
change_date_sto = change_date_sto.set_index('Unnamed: 0').to_dict()['0']
old_eq_sto=load_data('old_eq_sto.csv')
old_eq_sto.set_index('Unnamed: 0',inplace=True)
old_eq_sto.index = pd.to_datetime(old_eq_sto.index)
current_sto=load_data('current_sto.csv')
current_sto.set_index('Unnamed: 0',inplace=True)
current_sto_w=load_data('current_sto_w.csv')
current_sto_w.set_index('Unnamed: 0',inplace=True)

d_fo=load_data('d_fo.csv')
d_fo = d_fo.set_index('Unnamed: 0').to_dict()
win_pct_fo=load_data('win_pct_fo.csv')
win_pct_fo.set_index('Unnamed: 0',inplace=True)
frames_all_fo=load_data('frames_all_fo.csv')
frames_all_fo.set_index('Unnamed: 0',inplace=True)
frames_all_fo.index = pd.to_datetime(frames_all_fo.index)
summary_table_fo=load_data('summary_table_fo.csv')
summary_table_fo.set_index('Unnamed: 0',inplace=True)
change_date_fo=load_data('change_date_fo.csv')
change_date_fo = change_date_fo.set_index('Unnamed: 0').to_dict()['0']
old_eq_fo=load_data('old_eq_fo.csv')
old_eq_fo.set_index('Unnamed: 0',inplace=True)
old_eq_fo.index = pd.to_datetime(old_eq_fo.index)
current_fo=load_data('current_fo.csv')
current_fo.set_index('Unnamed: 0',inplace=True)
current_fo_w=load_data('current_fo_w.csv')
current_fo_w.set_index('Unnamed: 0',inplace=True)

d=load_data('d.csv')
d = d.set_index('Unnamed: 0').to_dict()
win_pct=load_data('win_pct.csv')
win_pct.set_index('Unnamed: 0',inplace=True)
frames_all=load_data('frames_all.csv')
frames_all.set_index('Unnamed: 0',inplace=True)
frames_all.index = pd.to_datetime(frames_all.index)
summary_table=load_data('summary_table.csv')
summary_table.set_index('Unnamed: 0',inplace=True)
change_date=load_data('change_date.csv')
change_date = change_date.set_index('Unnamed: 0').to_dict()['0']
old_eq=load_data('old_eq.csv')
old_eq.set_index('Unnamed: 0',inplace=True)
old_eq.index = pd.to_datetime(old_eq.index)
current=load_data('current.csv')
current.set_index('Unnamed: 0',inplace=True)
current_w=load_data('current_w.csv')
current_w.set_index('Unnamed: 0',inplace=True)

symbols = summary_table['symbol'].unique()

# Initialize Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("SpeedLab Monitoring Dashboard"),
    html.Div([
        html.Label("Select Dataset:"),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[
                {'label': 'Indices', 'value': 'ind'},
                {'label': 'Stocks', 'value': 'sto'},
                {'label': 'Crypto', 'value': 'cry'},
                {'label': 'Forex', 'value': 'forex'}
            ],
            value='cry'
        )
    ], style={'width': '50%', 'margin': '0 auto'}),
    html.Div([
        html.Label("Select Symbol:"),
        dcc.Dropdown(
            id='symbol-dropdown',
            options=[{'label': symbol, 'value': symbol} for symbol in symbols],
            value=symbols[0]
        )
    ], style={'width': '50%', 'margin': '0 auto'}),

    # Graph for cumsum pnl
    dcc.Graph(id='pnl-plot'),

    # Current table
    html.Div([
        html.H2("Current Week"),
        dash_table.DataTable(
            id='current-week-table',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'}
        )
    ]),

    # Current table
    html.Div([
        html.H2("Current Month"),
        dash_table.DataTable(
            id='current-table',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'}
        )
    ]),

    # Summary table
    html.Div([
        html.H2("Summary Table"),
        dash_table.DataTable(
            id='summary-table',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'}
        )
    ]),

    # Win percent table
    html.Div([
        html.H2("Win Percent Table"),
        dash_table.DataTable(
            id='win-percent-table',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'}
        )
    ])
])


@app.callback(
    [Output('symbol-dropdown', 'options'),
     Output('symbol-dropdown', 'value')],
    Input('dataset-dropdown', 'value')
)
def update_symbols(selected_dataset):
    if selected_dataset == 'ind':
        symbols = summary_table['symbol'].unique()
    elif selected_dataset == 'sto':
        symbols = list(summary_table_sto['symbol'].unique())
        symbols.append('total')
    elif selected_dataset == 'cry':
        symbols = list(summary_table_cry['symbol'].unique())
        symbols.append('total')
    else:
        symbols = summary_table_fo['symbol'].unique()

    options = [{'label': symbol, 'value': symbol} for symbol in symbols]
    default_value = symbols[0] if len(symbols) > 0 else None
    return options, default_value


@app.callback(
    [Output('pnl-plot', 'figure'),
     Output('current-week-table', 'data'),
     Output('current-week-table', 'columns'),
     Output('current-week-table', 'style_data_conditional'),
     Output('current-table', 'data'),
     Output('current-table', 'columns'),
     Output('current-table', 'style_data_conditional'),
     Output('summary-table', 'data'),
     Output('summary-table', 'columns'),
     Output('win-percent-table', 'data'),
     Output('win-percent-table', 'columns')],
    [Input('dataset-dropdown', 'value'),
     Input('symbol-dropdown', 'value')]
)
def update_dashboard(selected_dataset, selected_symbol):
    if selected_dataset == 'ind':
        change = pd.to_datetime(change_date[selected_symbol])
        before_date = frames_all.loc[:change]
        after_date = frames_all.loc[change:]
        before_date_old_eq = old_eq.loc[:change]
        after_date_old_eq = old_eq.loc[change:]
        trace_before = go.Scatter(
            x=before_date.index,
            y=before_date[selected_symbol],
            mode='lines',
            name='Backtest',
            line=dict(color='blue')
        )

        trace_after = go.Scatter(
            x=after_date.index,
            y=after_date[selected_symbol],
            mode='lines',
            name='Live',
            line=dict(color='orange')
        )
        # trace_before_old_eq = go.Scatter(
        #     x=before_date_old_eq.index,
        #     y=before_date_old_eq[selected_symbol],
        #     mode='lines',
        #     name='Old Equity Before Change Date',
        #     line=dict(color='green')
        # )

        trace_after_old_eq = go.Scatter(
            x=after_date_old_eq.index,
            y=after_date_old_eq[selected_symbol],
            mode='lines',
            name='Backtest After',
            line=dict(color='red')
        )

        fig = go.Figure(data=[trace_before, trace_after, trace_after_old_eq])
        # fig.add_trace(go.Scatter(
        #     x=frames_all.index,
        #     y=frames_all[selected_symbol],
        #     mode='lines',
        #     name=selected_symbol
        # ))
        fig.update_layout(
            title=f'Cumulative PnL for {selected_symbol}',
            xaxis_title='Date',
            yaxis_title='Cumulative PnL',
            # template='plotly_dark'
        )
        # Update current table
        current_data = pd.DataFrame(current.loc[selected_symbol]).T
        current_columns = [{'name': col, 'id': col} for col in current.columns]

        # Add conditional formatting for alerts
        alert_value = current_data['Alert'].iloc[0]
        alert_value2 = summary_table[summary_table['symbol'] == selected_symbol]['Underwater (days)'].iloc[0]
        style_data_conditional = []
        if current_data['Current Month'].iloc[0] < alert_value:
            style_data_conditional.append({
                'if': {
                    'filter_query': '{{Current Month}} < {:.2f}'.format(alert_value),
                    'column_id': 'Current Month'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })
        if current_data['Underwater'].iloc[0] > alert_value2:
            style_data_conditional.append({
                'if': {
                    'filter_query': '{{Underwater}} > {:.2f}'.format(alert_value2),
                    'column_id': 'Underwater'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })

        current_data_w = pd.DataFrame(current_w.loc[selected_symbol]).T
        current_columns_w = [{'name': col, 'id': col} for col in current_w.columns]

        # Add conditional formatting for alerts
        alert_value = current_data_w['Alert'].iloc[0]
        alert_value2 = summary_table[summary_table['symbol'] == selected_symbol]['Underwater (days)'].iloc[0]
        style_data_conditional_w = []
        if current_data_w['Current Week'].iloc[0] < alert_value:
            style_data_conditional_w.append({
                'if': {
                    'filter_query': '{{Current Week}} < {:.2f}'.format(alert_value),
                    'column_id': 'Current Week'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })
        if current_data_w['Underwater'].iloc[0] > alert_value2:
            style_data_conditional_w.append({
                'if': {
                    'filter_query': '{{Underwater}} > {:.2f}'.format(alert_value2),
                    'column_id': 'Underwater'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })

        # Update summary table
        summary_data = summary_table[summary_table['symbol'] == selected_symbol]
        summary_columns = [{'name': col, 'id': col} for col in summary_data.columns]

        # Update win percent table
        win_data = win_pct[selected_symbol].reset_index()
        win_columns = [{'name': col, 'id': col} for col in win_data.columns]

        return fig, current_data_w.to_dict(
            'records'), current_columns_w, style_data_conditional_w, current_data.to_dict(
            'records'), current_columns, style_data_conditional, summary_data.to_dict(
            'records'), summary_columns, win_data.to_dict('records'), win_columns
    elif selected_dataset == 'forex':
        change = pd.to_datetime(change_date_fo[selected_symbol])
        before_date = frames_all_fo.loc[:change]
        after_date = frames_all_fo.loc[change:]
        before_date_old_eq = old_eq_fo.loc[:change]
        after_date_old_eq = old_eq_fo.loc[change:]
        trace_before = go.Scatter(
            x=before_date.index,
            y=before_date[selected_symbol],
            mode='lines',
            name='Backtest',
            line=dict(color='blue')
        )

        trace_after = go.Scatter(
            x=after_date.index,
            y=after_date[selected_symbol],
            mode='lines',
            name='Live',
            line=dict(color='orange')
        )
        # trace_before_old_eq = go.Scatter(
        #     x=before_date_old_eq.index,
        #     y=before_date_old_eq[selected_symbol],
        #     mode='lines',
        #     name='Old Equity Before Change Date',
        #     line=dict(color='green')
        # )

        trace_after_old_eq = go.Scatter(
            x=after_date_old_eq.index,
            y=after_date_old_eq[selected_symbol],
            mode='lines',
            name='Backtest After',
            line=dict(color='red')
        )

        fig = go.Figure(data=[trace_before, trace_after, trace_after_old_eq])
        # fig.add_trace(go.Scatter(
        #     x=frames_all.index,
        #     y=frames_all[selected_symbol],
        #     mode='lines',
        #     name=selected_symbol
        # ))
        fig.update_layout(
            title=f'Cumulative PnL for {selected_symbol}',
            xaxis_title='Date',
            yaxis_title='Cumulative PnL',
            # template='plotly_dark'
        )
        # Update current table
        current_data = pd.DataFrame(current_fo.loc[selected_symbol]).T
        current_columns = [{'name': col, 'id': col} for col in current_fo.columns]

        # Add conditional formatting for alerts
        alert_value = current_data['Alert'].iloc[0]
        alert_value2 = summary_table_fo[summary_table_fo['symbol'] == selected_symbol]['Underwater (days)'].iloc[0]
        style_data_conditional = []
        if current_data['Current Month'].iloc[0] < alert_value:
            style_data_conditional.append({
                'if': {
                    'filter_query': '{{Current Month}} < {:.2f}'.format(alert_value),
                    'column_id': 'Current Month'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })
        if current_data['Underwater'].iloc[0] > alert_value2:
            style_data_conditional.append({
                'if': {
                    'filter_query': '{{Underwater}} > {:.2f}'.format(alert_value2),
                    'column_id': 'Underwater'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })

        current_data_w = pd.DataFrame(current_fo_w.loc[selected_symbol]).T
        current_columns_w = [{'name': col, 'id': col} for col in current_fo_w.columns]

        # Add conditional formatting for alerts
        alert_value = current_data_w['Alert'].iloc[0]
        alert_value2 = summary_table_fo[summary_table_fo['symbol'] == selected_symbol]['Underwater (days)'].iloc[0]
        style_data_conditional_w = []
        if current_data_w['Current Week'].iloc[0] < alert_value:
            style_data_conditional_w.append({
                'if': {
                    'filter_query': '{{Current Week}} < {:.2f}'.format(alert_value),
                    'column_id': 'Current Week'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })
        if current_data_w['Underwater'].iloc[0] > alert_value2:
            style_data_conditional_w.append({
                'if': {
                    'filter_query': '{{Underwater}} > {:.2f}'.format(alert_value2),
                    'column_id': 'Underwater'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })

        # Update summary table
        summary_data = summary_table_fo[summary_table_fo['symbol'] == selected_symbol]
        summary_columns = [{'name': col, 'id': col} for col in summary_data.columns]

        # Update win percent table
        win_data = win_pct_fo[selected_symbol].reset_index()
        win_columns = [{'name': col, 'id': col} for col in win_data.columns]

        return fig, current_data_w.to_dict(
            'records'), current_columns_w, style_data_conditional_w, current_data.to_dict(
            'records'), current_columns, style_data_conditional, summary_data.to_dict(
            'records'), summary_columns, win_data.to_dict('records'), win_columns
    elif selected_dataset == 'cry':
        change = pd.to_datetime(change_date_cry[selected_symbol])
        before_date = frames_all_cry.loc[:change]
        after_date = frames_all_cry.loc[change:]
        before_date_old_eq = old_eq_cry.loc[:change]
        after_date_old_eq = old_eq_cry.loc[change:]
        trace_before = go.Scatter(
            x=before_date.index,
            y=before_date[selected_symbol],
            mode='lines',
            name='Backtest',
            line=dict(color='blue')
        )

        trace_after = go.Scatter(
            x=after_date.index,
            y=after_date[selected_symbol],
            mode='lines',
            name='Live',
            line=dict(color='orange')
        )
        # trace_before_old_eq = go.Scatter(
        #     x=before_date_old_eq.index,
        #     y=before_date_old_eq[selected_symbol],
        #     mode='lines',
        #     name='Old Equity Before Change Date',
        #     line=dict(color='green')
        # )

        trace_after_old_eq = go.Scatter(
            x=after_date_old_eq.index,
            y=after_date_old_eq[selected_symbol],
            mode='lines',
            name='Backtest After',
            line=dict(color='red')
        )

        fig = go.Figure(data=[trace_before, trace_after, trace_after_old_eq])
        # fig.add_trace(go.Scatter(
        #     x=frames_all.index,
        #     y=frames_all[selected_symbol],
        #     mode='lines',
        #     name=selected_symbol
        # ))
        fig.update_layout(
            title=f'Cumulative PnL for {selected_symbol}',
            xaxis_title='Date',
            yaxis_title='Cumulative PnL',
            # template='plotly_dark'
        )
        # Update current table
        current_data = pd.DataFrame(current_cry.loc[selected_symbol]).T
        current_columns = [{'name': col, 'id': col} for col in current_cry.columns]

        # Add conditional formatting for alerts
        alert_value = current_data['Alert'].iloc[0]
        alert_value2 = summary_table_cry[summary_table_cry['symbol'] == selected_symbol]['Underwater (days)'].iloc[0]
        style_data_conditional = []
        if current_data['Current Month'].iloc[0] < alert_value:
            style_data_conditional.append({
                'if': {
                    'filter_query': '{{Current Month}} < {:.2f}'.format(alert_value),
                    'column_id': 'Current Month'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })
        if current_data['Underwater'].iloc[0] > alert_value2:
            style_data_conditional.append({
                'if': {
                    'filter_query': '{{Underwater}} > {:.2f}'.format(alert_value2),
                    'column_id': 'Underwater'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })

        current_data_w = pd.DataFrame(current_cry_w.loc[selected_symbol]).T
        current_columns_w = [{'name': col, 'id': col} for col in current_cry_w.columns]

        # Add conditional formatting for alerts
        alert_value = current_data_w['Alert'].iloc[0]
        alert_value2 = summary_table_cry[summary_table_cry['symbol'] == selected_symbol]['Underwater (days)'].iloc[0]
        style_data_conditional_w = []
        if current_data_w['Current Week'].iloc[0] < alert_value:
            style_data_conditional_w.append({
                'if': {
                    'filter_query': '{{Current Week}} < {:.2f}'.format(alert_value),
                    'column_id': 'Current Week'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })
        if current_data_w['Underwater'].iloc[0] > alert_value2:
            style_data_conditional_w.append({
                'if': {
                    'filter_query': '{{Underwater}} > {:.2f}'.format(alert_value2),
                    'column_id': 'Underwater'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })

        # Update summary table
        summary_data = summary_table_cry[summary_table_cry['symbol'] == selected_symbol]
        summary_columns = [{'name': col, 'id': col} for col in summary_data.columns]

        # Update win percent table
        win_data = win_pct_cry[selected_symbol].reset_index()
        win_columns = [{'name': col, 'id': col} for col in win_data.columns]

        return fig, current_data_w.to_dict(
            'records'), current_columns_w, style_data_conditional_w, current_data.to_dict(
            'records'), current_columns, style_data_conditional, summary_data.to_dict(
            'records'), summary_columns, win_data.to_dict('records'), win_columns
    else:
        change = pd.to_datetime(change_date_sto[selected_symbol])
        before_date = frames_all_sto.loc[:change]
        after_date = frames_all_sto.loc[change:]
        before_date_old_eq = old_eq_sto.loc[:change]
        after_date_old_eq = old_eq_sto.loc[change:]
        trace_before = go.Scatter(
            x=before_date.index,
            y=before_date[selected_symbol],
            mode='lines',
            name='Backtest',
            line=dict(color='blue')
        )

        trace_after = go.Scatter(
            x=after_date.index,
            y=after_date[selected_symbol],
            mode='lines',
            name='Live',
            line=dict(color='orange')
        )
        # trace_before_old_eq = go.Scatter(
        #     x=before_date_old_eq.index,
        #     y=before_date_old_eq[selected_symbol],
        #     mode='lines',
        #     name='Old Equity Before Change Date',
        #     line=dict(color='green')
        # )

        trace_after_old_eq = go.Scatter(
            x=after_date_old_eq.index,
            y=after_date_old_eq[selected_symbol],
            mode='lines',
            name='Backtest After',
            line=dict(color='red')
        )

        fig = go.Figure(data=[trace_before, trace_after, trace_after_old_eq])
        # fig.add_trace(go.Scatter(
        #     x=frames_all.index,
        #     y=frames_all[selected_symbol],
        #     mode='lines',
        #     name=selected_symbol
        # ))
        fig.update_layout(
            title=f'Cumulative PnL for {selected_symbol}',
            xaxis_title='Date',
            yaxis_title='Cumulative PnL',
            # template='plotly_dark'
        )
        # Update current table
        current_data = pd.DataFrame(current_sto.loc[selected_symbol]).T
        current_columns = [{'name': col, 'id': col} for col in current_sto.columns]

        # Add conditional formatting for alerts
        alert_value = current_data['Alert'].iloc[0]
        alert_value2 = summary_table_sto[summary_table_sto['symbol'] == selected_symbol]['Underwater (days)'].iloc[0]
        style_data_conditional = []
        if current_data['Current Month'].iloc[0] < alert_value:
            style_data_conditional.append({
                'if': {
                    'filter_query': '{{Current Month}} < {:.2f}'.format(alert_value),
                    'column_id': 'Current Month'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })
        if current_data['Underwater'].iloc[0] > alert_value2:
            style_data_conditional.append({
                'if': {
                    'filter_query': '{{Underwater}} > {:.2f}'.format(alert_value2),
                    'column_id': 'Underwater'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })

        current_data_w = pd.DataFrame(current_sto_w.loc[selected_symbol]).T
        current_columns_w = [{'name': col, 'id': col} for col in current_sto_w.columns]

        # Add conditional formatting for alerts
        alert_value = current_data_w['Alert'].iloc[0]
        alert_value2 = summary_table_sto[summary_table_sto['symbol'] == selected_symbol]['Underwater (days)'].iloc[0]
        style_data_conditional_w = []
        if current_data_w['Current Week'].iloc[0] < alert_value:
            style_data_conditional_w.append({
                'if': {
                    'filter_query': '{{Current Week}} < {:.2f}'.format(alert_value),
                    'column_id': 'Current Week'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })
        if current_data_w['Underwater'].iloc[0] > alert_value2:
            style_data_conditional_w.append({
                'if': {
                    'filter_query': '{{Underwater}} > {:.2f}'.format(alert_value2),
                    'column_id': 'Underwater'
                },
                'backgroundColor': 'red',
                'color': 'white'
            })

        # Update summary table
        summary_data = summary_table_sto[summary_table_sto['symbol'] == selected_symbol]
        summary_columns = [{'name': col, 'id': col} for col in summary_data.columns]

        # Update win percent table
        win_data = win_pct_sto[selected_symbol].reset_index()
        win_columns = [{'name': col, 'id': col} for col in win_data.columns]

        return fig, current_data_w.to_dict(
            'records'), current_columns_w, style_data_conditional_w, current_data.to_dict(
            'records'), current_columns, style_data_conditional, summary_data.to_dict(
            'records'), summary_columns, win_data.to_dict('records'), win_columns


if __name__ == "__main__":
    app.run_server(debug=True)
