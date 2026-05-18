"""
Calculate accuracy of day, month, and year predictions for all 4 models
"""
import re
from collections import defaultdict
from datetime import datetime

# Month name to number mapping
MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}

# Day name to weekday mapping (0=Monday, 6=Sunday)
DAY_MAP = {
    'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4, 'SAT': 5, 'SUN': 6
}

def parse_prediction_line(line):
    """Extract input and prediction from a line"""
    # Format: [DAY] [MONTH] [LEAP] [YEAR] DD-MM-YYYY
    match = re.match(r'\[(.*?)\]\s+\[(.*?)\]\s+\[(.*?)\]\s+\[(.*?)\]\s+([\d\-]+)', line)
    if match:
        day_name, month_name, leap, year_code, date_str = match.groups()
        try:
            day_pred, month_pred, year_pred = date_str.split('-')
            return {
                'day_name': day_name,
                'month_name': month_name,
                'leap': leap,
                'year_code': int(year_code),
                'day_pred': int(day_pred),
                'month_pred': int(month_pred),
                'year_pred': int(year_pred)
            }
        except ValueError:
            return None
    return None

def is_valid_date(day, month, year):
    """Check if a date is valid"""
    try:
        datetime(year, month, day)
        return True
    except:
        return False

def get_weekday(day, month, year):
    """Get weekday (0=Mon, 6=Sun) for a date"""
    try:
        dt = datetime(year, month, day)
        return dt.weekday()
    except:
        return None

def calculate_model_accuracy(filename):
    """Calculate accuracy metrics for a model"""
    month_correct = 0
    day_correct = 0
    total = 0
    skipped = 0
    
    details = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('Saved'):
                    continue
                
                parsed = parse_prediction_line(line)
                if not parsed:
                    skipped += 1
                    continue
                
                total += 1
                expected_month = MONTH_MAP.get(parsed['month_name'])
                expected_day = DAY_MAP.get(parsed['day_name'])
                
                # Month accuracy
                if parsed['month_pred'] == expected_month:
                    month_correct += 1
                
                # Day of week accuracy
                if is_valid_date(parsed['day_pred'], parsed['month_pred'], parsed['year_pred']):
                    pred_weekday = get_weekday(parsed['day_pred'], parsed['month_pred'], parsed['year_pred'])
                    if pred_weekday == expected_day:
                        day_correct += 1
                
                details.append(parsed)
    
    except FileNotFoundError:
        return None
    
    return {
        'total': total,
        'skipped': skipped,
        'month_accuracy': (month_correct / total * 100) if total > 0 else 0,
        'day_accuracy': (day_correct / total * 100) if total > 0 else 0,
        'details': details
    }

def calculate_model_accuracy(filename):
    """Calculate accuracy metrics for a model"""
    month_correct = 0
    day_correct = 0
    total = 0
    skipped = 0
    
    details = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('Saved'):
                    continue
                
                parsed = parse_prediction_line(line)
                if not parsed:
                    skipped += 1
                    continue
                
                total += 1
                expected_month = MONTH_MAP.get(parsed['month_name'])
                expected_day = DAY_MAP.get(parsed['day_name'])
                
                # Month accuracy
                if parsed['month_pred'] == expected_month:
                    month_correct += 1
                
                # Day of week accuracy
                if is_valid_date(parsed['day_pred'], parsed['month_pred'], parsed['year_pred']):
                    pred_weekday = get_weekday(parsed['day_pred'], parsed['month_pred'], parsed['year_pred'])
                    if pred_weekday == expected_day:
                        day_correct += 1
                
                details.append(parsed)
    
    except FileNotFoundError:
        return None
    
    return {
        'total': total,
        'skipped': skipped,
        'month_accuracy': (month_correct / total * 100) if total > 0 else 0,
        'day_accuracy': (day_correct / total * 100) if total > 0 else 0,
        'details': details
    }

# Calculate for all 4 models
models = ['lstm', 'vae', 'cgan', 'diffusion']
results = {}

print("=" * 70)
print("ACCURACY REPORT - DAY, MONTH, YEAR PREDICTIONS")
print("=" * 70)

for model in models:
    filename = f'predictions_{model}.txt'
    accuracy = calculate_model_accuracy(filename)
    
    if accuracy:
        results[model] = accuracy
        print(f"\n{model.upper()} Model:")
        print(f"  Total predictions: {accuracy['total']}")
        if accuracy['skipped'] > 0:
            print(f"  Skipped (corrupted): {accuracy['skipped']}")
        print(f"  Month accuracy: {accuracy['month_accuracy']:.2f}%")
        print(f"  Day-of-week accuracy: {accuracy['day_accuracy']:.2f}%")
    else:
        print(f"\n{model.upper()}: File not found")

# Summary comparison
print("\n" + "=" * 70)
print("SUMMARY COMPARISON")
print("=" * 70)
print(f"{'Model':<12} {'Month Accuracy':<20} {'Day Accuracy':<20}")
print("-" * 70)
for model in models:
    if model in results:
        r = results[model]
        print(f"{model:<12} {r['month_accuracy']:>6.2f}%{' ' * 10} {r['day_accuracy']:>6.2f}%")

# Detailed analysis - show which months have errors
print("\n" + "=" * 70)
print("DETAILED ANALYSIS - DAY ACCURACY BY MONTH")
print("=" * 70)

for model in models:
    if model not in results:
        continue
    
    month_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for detail in results[model]['details']:
        expected_day = DAY_MAP.get(detail['day_name'])
        if is_valid_date(detail['day_pred'], detail['month_pred'], detail['year_pred']):
            pred_weekday = get_weekday(detail['day_pred'], detail['month_pred'], detail['year_pred'])
            month_stats[detail['month_name']]['total'] += 1
            if pred_weekday == expected_day:
                month_stats[detail['month_name']]['correct'] += 1
    
    print(f"\n{model.upper()}:")
    print(f"  {'Month':<8} {'Correct':<10} {'Total':<10} {'Day Accuracy':<12}")
    for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
        if month in month_stats:
            stats = month_stats[month]
            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {month:<8} {stats['correct']:<10} {stats['total']:<10} {acc:>6.2f}%")

print("\n" + "=" * 70)
