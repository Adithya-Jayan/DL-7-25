import numpy as np
import matplotlib.pyplot as plt

def calculate_accuracy_metrics(actual, predicted, model_name="Model"):
    """
    Calculate various accuracy metrics for model evaluation, with dynamic labels per model name.

    Parameters:
        actual (array-like): Actual values.
        predicted (array-like): Predicted values.
        model_name (str): Name of the model (e.g., "ARIMAX", "XGBoost", etc.).

    Returns:
        dict: A dictionary with accuracy metrics labeled by model.
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    actual = np.array(actual)
    predicted = np.array(predicted)

    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        return {f'{model_name}_Error': 'No valid predictions to evaluate'}

    # Core Metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    mse = mean_squared_error(actual, predicted)
    r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

    # Directional accuracy (up/down prediction)
    if len(actual) > 1:
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    else:
        directional_accuracy = 0

    """ return {
        f'{model_name}_MAE': mae,
        f'{model_name}_RMSE': rmse,
        f'{model_name}_MAPE': mape,
        f'{model_name}_MSE': mse,
        f'{model_name}_R²': r2,
        f'{model_name}_Directional_Accuracy': directional_accuracy,
        f'{model_name}_Sample_Size': len(actual)
    } """
    return {
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape,
    'MSE': mse,
    'R²': r2,
    'Directional_Accuracy': directional_accuracy,
    'Sample_Size': len(actual)
    }



def plot_model_results(results, model_name="ARIMAX", save_path=None):
    """
    Create 4-in-1 plot layout to visualize prediction performance for any model.

    Parameters:
        results (dict): Dictionary containing 'actuals', 'predictions', and optionally 'dates'.
        model_name (str): Name of the model (used in titles).
        save_path (str): Optional path to save the plot image.
    """
    if not results or len(results.get('predictions', [])) == 0:
        print("❌ No results to plot.")
        return

    actuals = results['actuals']
    predictions = results['predictions']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{model_name} Model Accuracy Test Results", fontsize=16)

    # Plot 1: Actual vs Predicted Prices
    axes[0, 0].plot(range(len(actuals)), actuals, 'b-', label='Actual Prices', linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(range(len(predictions)), predictions, 'r--', label='Predicted Prices', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_title('Actual vs Predicted Prices')
    axes[0, 0].set_xlabel('Test Day')
    axes[0, 0].set_ylabel('Price (₹)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Scatter Plot
    axes[0, 1].scatter(actuals, predictions, alpha=0.7, s=50)
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    axes[0, 1].set_xlabel('Actual Price (₹)')
    axes[0, 1].set_ylabel('Predicted Price (₹)')
    axes[0, 1].set_title('Scatter Plot: Actual vs Predicted')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Prediction Errors Over Time
    errors = np.array(predictions) - np.array(actuals)
    axes[1, 0].plot(range(len(errors)), errors, 'g-', linewidth=2, marker='D', markersize=3)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].fill_between(range(len(errors)), errors, 0, alpha=0.3, color='green')
    axes[1, 0].set_title('Prediction Errors Over Time')
    axes[1, 0].set_xlabel('Test Day')
    axes[1, 0].set_ylabel('Error (Predicted - Actual) ₹')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Error Distribution
    axes[1, 1].hist(errors, bins=min(15, len(errors) // 2 + 1), alpha=0.7,
                   color='skyblue', edgecolor='black', density=True)
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 1].axvline(x=np.mean(errors), color='orange', linestyle=':', linewidth=2, label='Mean Error')
    axes[1, 1].set_title('Distribution of Prediction Errors')
    axes[1, 1].set_xlabel('Error (Predicted - Actual) ₹')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust for suptitle

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")

    plt.show()





def print_model_metrics(results, model_name="ARIMAX"):
    """
    Print detailed accuracy and evaluation metrics for any model.

    Parameters:
        results (dict): Dictionary containing keys like 'metrics', 'actuals', 'predictions', etc.
        model_name (str): Name of the model (used in titles and labels).
    """
    if not results or len(results.get('predictions', [])) == 0:
        print("❌ No results to display")
        return

    # Title header
    print("\n" + "="*70)
    print(f"{model_name.upper()} MODEL - ACCURACY TEST RESULTS")
    print("="*70)

    metrics = results.get('metrics', {})
    test_period = results.get('test_period', len(results['actuals']))
    successful_predictions = results.get('successful_predictions', len(results['predictions']))
    actuals = np.array(results['actuals'])
    predicted = np.array(results['predictions'])
    errors = predicted - actuals


    print(f"Test Period: {test_period} days")
    print(f"Successful Predictions: {successful_predictions}/{test_period}")
    print(f"Success Rate: {(successful_predictions / test_period) * 100:.1f}%")

    # Accuracy metrics
    print(f"\n📊 ACCURACY METRICS:")
    print(f"├── Mean Absolute Error (MAE): ₹{metrics.get('MAE', 0):.2f}")
    print(f"├── Root Mean Squared Error (RMSE): ₹{metrics.get('RMSE', 0):.2f}")
    print(f"├── Mean Absolute Percentage Error (MAPE): {metrics.get('MAPE', 0):.2f}%")
    print(f"├── Mean Squared Error (MSE): {metrics.get('MSE', 0):.2f}")
    print(f"├── R-squared (R²): {metrics.get('R²', 0):.4f}")
    print(f"└── Directional Accuracy: {metrics.get('Directional_Accuracy', 0):.2f}%")

    # Price analysis
    print(f"\n💰 PRICE ANALYSIS:")
    print(f"├── Average Actual Price: ₹{np.mean(actuals):.2f}")
    print(f"├── Average Predicted Price: ₹{np.mean(predicted):.2f}")
    print(f"├── Average Bias: ₹{np.mean(predicted - actuals):.2f}")
    print(f"├── Largest Overestimate: ₹{np.max(errors):.2f}")
    print(f"└── Largest Underestimate: ₹{np.min(errors):.2f}")

    """ # Performance summary
    mape = metrics.get('MAPE', float('inf'))
    print(f"\n🎯 PERFORMANCE RATING:")
    if mape < 2:
        print("🏆 EXCELLENT - Model is highly accurate!")
    elif mape < 5:
        print("📈 VERY GOOD - Model performs well!")
    elif mape < 10:
        print("📊 GOOD - Model is reasonably accurate!")
    elif mape < 20:
        print("⚠️  FAIR - Needs improvement.")
    else:
        print("❌ POOR - Consider tuning the model or features.")

    print("="*70) """


    # Enhanced Performance Evaluation based on MAPE and Directional Accuracy
    print(f"\n🎯 PERFORMANCE EVALUATION:")
    print("-" * 40)

    mape = metrics.get('MAPE', float('inf'))
    r2 = metrics.get('R²', 0)
    directional_acc = metrics.get('Directional_Accuracy', 0)

    # Price Accuracy Rating
    if mape < 2:
        price_rating = "Excellent"
    elif mape < 5:
        price_rating = "Good"
    elif mape < 10:
        price_rating = "Fair"
    else:
        price_rating = "Poor"

    # Direction Accuracy Rating
    if directional_acc > 70:
        direction_rating = "Excellent"
    elif directional_acc > 60:
        direction_rating = "Good"
    elif directional_acc > 50:
        direction_rating = "Fair"
    else:
        direction_rating = "Poor"

    print(f"Price Accuracy Rating    : {price_rating} (MAPE: {mape:.2f}%)")
    print(f"Direction Accuracy Rating: {direction_rating} ({directional_acc:.1f}%)")
    print(f"Model Fit (R²)          : {r2:.4f}")



def print_model_comparison_table(results_dict):
    """
    Prints a detailed comparison table of evaluation metrics and predictions 
    for multiple models, including price prediction stats.
    """
    print("\n📊 MODEL COMPARISON TABLE (Last Prediction Snapshot)")
    print("─" * 130)
    header = f"{'Model':<15} {'Current (₹)':<13} {'Predicted (₹)':<15} {'Bias (₹)':<10} {'% Change':<10} {'MAE':<8} {'RMSE':<8} {'MAPE (%)':<10} {'R²':<8} {'DirAcc (%)':<10}"
    print(header)
    print("─" * 130)

    for model_name, result in results_dict.items():
        metrics = result.get('metrics', {})
        
        # Try nested (like Random Forest/XGBoost) or fallback to flat (like LSTM)
        if 'Price Metrics' in metrics:
            price = metrics['Price Metrics']
            direction = metrics.get('Percentage Change Metrics', {})
        else:
            price = metrics
            direction = metrics

        # Accuracy metrics
        mae = price.get('MAE', 0)
        rmse = price.get('RMSE', 0)
        mape = price.get('MAPE (%)', 0)
        r2 = price.get('R² Score', 0)
        dir_acc = direction.get('Directional Accuracy (%)', 0)

        # Price comparison
        current_price = result['actuals'][-1] if 'actuals' in result else 0
        predicted_price = result['predictions'][-1] if 'predictions' in result else 0
        bias = predicted_price - current_price
        percent_change = (bias / current_price) * 100 if current_price != 0 else 0

        row = f"{model_name:<15} ₹{current_price:<12.2f} ₹{predicted_price:<14.2f} ₹{bias:<9.2f} {percent_change:<9.2f}% {mae:<8.2f} {rmse:<8.2f} {mape:<10.2f} {r2:<8.4f} {dir_acc:<10.2f}"
        print(row)

    print("─" * 130)