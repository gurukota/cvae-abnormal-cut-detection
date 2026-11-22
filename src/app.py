"""Desktop interface for training and reviewing the VM20i experiment."""

import csv
import json
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from statistics import mean
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
METHODS = {
    'Real only': 'real_only', 'CVAE': 'cvae', 'SMOTE': 'smote',
    'Reconstruction': 'reconstruction', 'Jitter': 'jitter',
    'Neighbour interpolation': 'neighbour_interpolation',
    'Random duplication': 'random_duplication',
}
DETECTORS = {'Random Forest': 'rf', 'XGBoost': 'xgb'}
PARTITIONS = {'ID test': 'id_test', 'OOD test': 'ood_test'}
KEYS = ('detector', 'partition', 'fold', 'seed', 'arm')


def read_rows(folder, name):
    with (folder / name).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def completed_runs(folder):
    runs = {}
    for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
        try:
            completion = json.loads((path / 'completion.json').read_text())
        except (OSError, ValueError):
            continue
        if completion.get('complete') is True:
            runs[path.name] = completion
    return runs


def make_table(parent, columns, height):
    table = ttk.Treeview(parent, columns=[title for title, _ in columns],
                         show='headings', height=height)
    for title, width in columns:
        table.heading(title, text=title)
        table.column(title, width=width, minwidth=55,
                     anchor='w' if title == columns[0][0] else 'center')
    table.pack(fill='x', pady=(8, 0))
    return table


def fill_table(table, rows):
    table.delete(*table.get_children())
    for row in rows:
        table.insert('', 'end', values=row)


def percent(value):
    return f'{100 * float(value):.1f}%'


class App:
    def __init__(self, window, results=RESULTS):
        self.window = window
        self.results = results
        self.events = queue.Queue()
        self.training = False
        self.runs = {}
        self.summaries = []
        self.counts = []
        self.quality = []
        self.comparisons = []
        self.predictions = defaultdict(dict)
        self.thresholds = {}
        window.title('VM20i · Abnormal-cut detection')
        window.geometry('940x555')
        window.minsize(845, 500)
        window.protocol('WM_DELETE_WINDOW', self.close)

        body = ttk.Frame(window, padding=16)
        body.pack(fill='both', expand=True)
        header = ttk.Frame(body)
        header.pack(fill='x', pady=(0, 12))
        ttk.Label(header, text='VM20i experiment', font=('Segoe UI', 16)).pack(side='left')
        self.run = self.select(header, 'Results from', [], '', self.load_run, width=22, side='right')

        self.tabs = ttk.Notebook(body)
        self.tabs.pack(fill='both', expand=True)
        train = ttk.Frame(self.tabs, padding=14)
        results_tab = ttk.Frame(self.tabs, padding=14)
        inspect = ttk.Frame(self.tabs, padding=14)
        self.tabs.add(train, text='Train')
        self.tabs.add(results_tab, text='Results')
        self.tabs.add(inspect, text='Inspect a cut')
        self.build_training(train)
        self.build_results(results_tab)
        self.build_inspection(inspect)
        self.tabs.select(results_tab)

        ttk.Separator(body).pack(fill='x', pady=(9, 6))
        self.status = tk.StringVar()
        ttk.Label(body, textvariable=self.status).pack(anchor='w')
        self.refresh_runs('final')
        window.after(100, self.poll_training)

    def select(self, parent, title, values, initial, callback, width=17, side='left'):
        field = ttk.Frame(parent)
        field.pack(side=side, padx=(0, 12))
        ttk.Label(field, text=title).pack(anchor='w')
        combo = ttk.Combobox(field, values=values, state='readonly', width=width)
        combo.set(initial)
        combo.pack(anchor='w')
        combo.bind('<<ComboboxSelected>>', lambda _: callback())
        return combo

    def build_training(self, parent):
        controls = ttk.Frame(parent)
        controls.pack(fill='x')
        name = ttk.Frame(controls)
        name.pack(side='left', padx=(0, 12))
        ttk.Label(name, text='New run name').pack(anchor='w')
        self.run_name = ttk.Entry(name, width=23)
        self.run_name.insert(0, 'run')
        self.run_name.pack()
        self.mode = self.select(controls, 'Training mode', ['Full run', 'Quick check'],
                                'Full run', lambda: None)
        self.start = ttk.Button(controls, text='Start training', command=self.start_training)
        self.start.pack(side='left', anchor='s')
        ttk.Label(parent, text='A quick check uses one seed and fewer epochs and trees.').pack(
            anchor='w', pady=(10, 8))
        log_frame = ttk.Frame(parent)
        log_frame.pack(fill='both', expand=True)
        self.log = tk.Text(log_frame, height=10, state='disabled', wrap='word',
                           font=('Consolas', 10), padx=8, pady=8)
        scroll = ttk.Scrollbar(log_frame, orient='vertical', command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.log.pack(side='left', fill='both', expand=True)

    def build_results(self, parent):
        views = ttk.Notebook(parent)
        views.pack(fill='both', expand=True)
        detection = ttk.Frame(views, padding=8)
        quality = ttk.Frame(views, padding=8)
        comparisons = ttk.Frame(views, padding=8)
        views.add(detection, text='Detection')
        views.add(quality, text='Feature quality')
        views.add(comparisons, text='CVAE comparisons')
        self.comparison_detector = self.select(comparisons, 'Classifier', list(DETECTORS), 'Random Forest', self.show_comparisons)
        self.comparison_partition = self.select(comparisons, 'Test partition', list(PARTITIONS), 'OOD test', self.show_comparisons)
        self.comparison_table = make_table(comparisons, [
            ('Compared with', 170), ('Recall change (pp)', 125),
            ('FAR change (pp)', 115), ('Interpretation', 350),
        ], 6)
        ttk.Label(comparisons, text='Descriptive mean paired differences. Rule added after the original results were inspected.').pack(anchor='w', pady=(8, 0))
        filters = ttk.Frame(detection)
        filters.pack(fill='x')
        self.detector = self.select(filters, 'Classifier', list(DETECTORS), 'Random Forest', self.show_results)
        self.partition = self.select(filters, 'Test partition', list(PARTITIONS), 'OOD test', self.show_results)
        self.scores_table = make_table(detection, [
            ('Training method', 220), ('Recall', 110), ('False-alarm rate', 120),
            ('Avg. precision', 120), ('Fits', 65),
        ], 7)
        self.count_text = tk.StringVar()
        ttk.Label(detection, textvariable=self.count_text).pack(anchor='w', pady=(9, 0))
        self.quality_table = make_table(quality, [
            ('Training method', 190), ('Variance vs real', 125), ('Mean gap', 95),
            ('Correlation gap', 115), ('Near copies', 100),
        ], 6)
        ttk.Label(quality, text='Compared with same-class validation cuts. Near copies are measured against training cuts.').pack(
            anchor='w', pady=(9, 0))

    def build_inspection(self, parent):
        filters = ttk.Frame(parent)
        filters.pack(fill='x')
        self.inspect_detector = self.select(filters, 'Classifier', list(DETECTORS), 'Random Forest', self.list_cuts)
        self.inspect_partition = self.select(filters, 'Test partition', list(PARTITIONS), 'OOD test', self.list_cuts)
        self.fold = self.select(filters, 'Fold', [], '', self.list_cuts, width=6)
        self.seed = self.select(filters, 'Seed', [], '', self.list_cuts, width=6)
        choices = ttk.Frame(parent)
        choices.pack(fill='x', pady=(12, 0))
        self.cut = self.select(choices, 'Recorded cut', [], '', self.show_cut, width=28)
        self.first = self.select(choices, 'Method A', list(METHODS), 'Real only', self.show_cut, width=19)
        self.second = self.select(choices, 'Method B', list(METHODS), 'CVAE', self.show_cut, width=19)
        self.annotation = tk.StringVar()
        ttk.Label(parent, textvariable=self.annotation).pack(anchor='w', pady=(15, 0))
        self.cut_table = make_table(parent, [
            ('Training method', 210), ('Score', 150), ('Alarm threshold', 150),
            ('Decision', 140),
        ], 2)

    def refresh_runs(self, preferred):
        self.runs = completed_runs(self.results)
        self.run.configure(values=list(self.runs))
        selected = preferred if preferred in self.runs else next(iter(self.runs), '')
        self.run.set(selected)
        self.load_run()

    def load_run(self):
        name = self.run.get()
        self.summaries, self.counts, self.quality = [], [], []
        self.comparisons = []
        self.metrics = []
        self.predictions.clear()
        self.thresholds.clear()
        folds, seeds = [], []
        self.status.set('No completed runs yet.')
        if name:
            folder = self.results / name
            try:
                self.summaries = read_rows(folder, 'summary.csv')
                self.counts = read_rows(folder, 'partition_counts.csv')
                self.quality = read_rows(folder, 'feature_quality.csv')
                metrics = read_rows(folder, 'metrics.csv')
                self.metrics = metrics
                if (folder / 'comparisons.csv').exists():
                    self.comparisons = read_rows(folder, 'comparisons.csv')
                self.thresholds = {tuple(row[key] for key in KEYS): row['threshold'] for row in metrics}
                for row in read_rows(folder, 'predictions.csv'):
                    key = tuple(row[key] for key in KEYS)
                    self.predictions[key][row['row_id']] = row
                folds = sorted({row['fold'] for row in metrics}, key=int)
                seeds = sorted({row['seed'] for row in metrics}, key=int)
                mode = 'Quick check' if self.runs[name].get('smoke_only') else 'Full run'
                self.status.set(f'{name} · {mode} · {len(folds)} folds · {len(seeds)} seeds')
            except (OSError, KeyError, ValueError) as error:
                self.status.set(f'Could not read {name}: {error}')
        for combo, values in ((self.fold, folds), (self.seed, seeds)):
            combo.configure(values=values)
            combo.set(values[0] if values else '')
        self.show_results()
        self.show_comparisons()
        self.list_cuts()

    def show_results(self):
        classifier = DETECTORS[self.detector.get()]
        partition = PARTITIONS[self.partition.get()]
        selected = {row['arm']: row for row in self.summaries
                    if row['detector'] == classifier and row['partition'] == partition}
        counts = defaultdict(int)
        for row in self.metrics:
            if row['detector'] == classifier and row['partition'] == partition:
                counts[row['arm']] += 1
        fill_table(self.scores_table, [
            (label, percent(row['recall']), percent(row['false_alarm_rate']),
             f"{float(row['average_precision']):.3f}", counts[arm])
            for label, arm in METHODS.items() if (row := selected.get(arm))
        ])
        counts = [f"fold {row['fold']}: {row['positive']} abnormal, {row['negative']} other"
                  for row in self.counts if row['partition'] == partition]
        self.count_text.set('Test cuts per fold · ' + '  |  '.join(counts) if counts else '')
        quality = defaultdict(list)
        for row in self.quality:
            quality[row['arm']].append(row)
        entries = []
        for label, arm in METHODS.items():
            if rows := quality.get(arm):
                real_variance = mean(float(row['validation_variance']) for row in rows)
                generated_variance = mean(float(row['variance']) for row in rows)
                entries.append((label, percent(generated_variance / real_variance) if real_variance else '—',
                                f"{mean(float(row['mean_shift']) for row in rows):.3f}",
                                f"{mean(float(row['correlation_error']) for row in rows):.3f}",
                                percent(mean(float(row['duplicate_fraction']) for row in rows))))
        fill_table(self.quality_table, entries)

    def show_comparisons(self):
        labels = {value: key for key, value in METHODS.items()}
        rows = [row for row in self.comparisons
                if row['detector'] == DETECTORS[self.comparison_detector.get()]
                and row['partition'] == PARTITIONS[self.comparison_partition.get()]]
        fill_table(self.comparison_table, [
            (labels.get(row['control'], row['control']),
             f"{100 * float(row['mean_recall_delta']):+.2f}",
             f"{100 * float(row['mean_false_alarm_rate_delta']):+.2f}",
             row['interpretation']) for row in rows
        ])

    def inspection_key(self, method):
        return (DETECTORS[self.inspect_detector.get()], PARTITIONS[self.inspect_partition.get()],
                self.fold.get(), self.seed.get(), METHODS[method])

    def list_cuts(self):
        cuts = list(self.predictions.get(self.inspection_key('Real only'), {}))
        current = self.cut.get()
        self.cut.configure(values=cuts)
        self.cut.set(current if current in cuts else next(iter(cuts), ''))
        self.show_cut()

    def show_cut(self):
        selected = self.cut.get()
        results = []
        annotation = ''
        for method in (self.first.get(), self.second.get()):
            key = self.inspection_key(method)
            row = self.predictions.get(key, {}).get(selected)
            if row:
                annotation = 'Annotated abnormal' if row['label'] == '1' else 'Annotated non-abnormal'
                results.append((method, row['score'], self.thresholds[key],
                                'Alarm' if row['alarm'] == 'True' else 'No alarm'))
        self.annotation.set(annotation)
        fill_table(self.cut_table, results)

    def append_log(self, line):
        self.log.configure(state='normal')
        self.log.insert('end', line)
        self.log.see('end')
        self.log.configure(state='disabled')

    def start_training(self):
        name = self.run_name.get().strip()
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}', name):
            messagebox.showerror('Run name', 'Use 1–40 letters, numbers, hyphens or underscores.')
            return
        output = self.results / name
        if output.exists():
            messagebox.showerror('Run name', 'That output folder already exists. Choose another name.')
            return
        command = [sys.executable, '-u', str(ROOT / 'src/train.py'), '--output', str(output)]
        if self.mode.get() == 'Quick check':
            command.append('--smoke')
        self.training = True
        self.start.state(['disabled'])
        self.append_log(f'\nStarting {name}…\n')
        threading.Thread(target=self.run_training, args=(command, name), daemon=True).start()

    def run_training(self, command, name):
        try:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors='replace',
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as process:
                for line in process.stdout:
                    self.events.put(('log', line))
                succeeded = process.wait() == 0
        except OSError as error:
            self.events.put(('log', f'{error}\n'))
            succeeded = False
        self.events.put(('finished', (name, succeeded)))

    def poll_training(self):
        while not self.events.empty():
            kind, value = self.events.get_nowait()
            if kind == 'log':
                self.append_log(value)
                continue
            name, succeeded = value
            self.training = False
            self.start.state(['!disabled'])
            if succeeded:
                self.refresh_runs(name)
                self.append_log('Finished. Open Results to review this run.\n')
            else:
                self.append_log('Training failed. See the message above.\n')
        self.window.after(100, self.poll_training)

    def close(self):
        if self.training:
            messagebox.showinfo('Training in progress', 'Keep this window open until the run finishes.')
        else:
            self.window.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
