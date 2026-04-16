import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# === KONFIGURACJA STYLU ===
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (16, 12)})

# === DEFINICJA SCENARIUSZY ===
SCENARIOS = {
    "Baseline": {
        "file": os.path.join("Baseline", "wyniki_baseline_single.csv"), 
        "color": "#d62728",
        "style": "--"
    },
    "Rondo": {
        "file": os.path.join("Baseline_rondo", "wyniki_rondo.csv"),
        "color": "#1f77b4",
        "style": "-."
    },
    "AI (1 Skrzyżowanie)": {
        "file": os.path.join("RL_DQL", "wyniki_AI_1_skrzyzowanie.csv"),
        "color": "#2ca02c",
        "style": "-"
    },
    "AI (2 Skrzyżowania)": {
        "file": os.path.join("RL_DQL_2", "wyniki_AI_2_skrzyzowania.csv"),
        "color": "#9467bd",
        "style": "-"
    }
}

def plot_results():
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    stats_summary = []
    
    emerg_times = {}

    print("--- WCZYTYWANIE DANYCH ---")
    print(f"Szukam plików w folderze roboczym: {os.getcwd()}")

    for name, config in SCENARIOS.items():
        filepath = config["file"]
        
        if not os.path.exists(filepath):
            print(f"⚠️  BŁĄD: Nie znaleziono pliku: {filepath}")
            print(f"    Upewnij się, że folder '{os.path.dirname(filepath)}' istnieje i zawiera plik .csv")
            continue
            
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Wczytano: {name} ({len(df)} wierszy)")

            # --- WYGŁADZANIE DANYCH  ---
            q_col = 'total_queue' if 'total_queue' in df.columns else 'queue'
            
            df['queue_smooth'] = df[q_col].rolling(window=50, min_periods=1).mean()
            df['wait_smooth'] = df['avg_wait_time'].rolling(window=50, min_periods=1).mean()

            axes[0, 0].plot(df['step'], df['queue_smooth'], 
                            label=name, color=config['color'], linestyle=config['style'], linewidth=2)

            axes[0, 1].plot(df['step'], df['wait_smooth'], 
                            label=name, color=config['color'], linestyle=config['style'], linewidth=2)

            if 'amb_time' in df.columns:
                amb = pd.to_numeric(df['amb_time'], errors='coerce').dropna()
                fire = pd.to_numeric(df['fire_time'], errors='coerce').dropna()
                all_emerg = pd.concat([amb, fire])
            else:
                amb = pd.to_numeric(df.get('ambulance_time', []), errors='coerce').dropna()
                fire = pd.to_numeric(df.get('firetruck_time', []), errors='coerce').dropna()
                all_emerg = pd.concat([amb, fire])
            
            avg_emerg_time = all_emerg.mean() if not all_emerg.empty else 0
            emerg_times[name] = avg_emerg_time

            stats_summary.append({
                "Scenariusz": name,
                "Śr. Kolejka": df[q_col].mean(),
                "Śr. Czas Oczekiwania": df['avg_wait_time'].mean(),
                "Śr. Czas Służb": avg_emerg_time
            })

        except Exception as e:
            print(f"❌ Błąd przy przetwarzaniu {name}: {e}")

    # --- FORMATOWANIE WYKRESU 1 (KOLEJKI) ---
    axes[0, 0].set_title('Średnia Długość Kolejki (Płynność)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Liczba pojazdów', fontsize=12)
    axes[0, 0].set_xlabel('Krok symulacji [s]', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # --- FORMATOWANIE WYKRESU 2 (WAIT TIME) ---
    axes[0, 1].set_title('Średni Czas Oczekiwania na Pojazd', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Czas [s]', fontsize=12)
    axes[0, 1].set_xlabel('Krok symulacji [s]', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # --- 3. WYKRES SŁUPKOWY - (CZAS SŁUŻB) ---
    gs = axes[1, 0].get_gridspec()
    for ax in axes[1, :]: ax.remove()
    ax_bar = fig.add_subplot(gs[1, :])

    if emerg_times:
        names = list(emerg_times.keys())
        values = list(emerg_times.values())
        colors = [SCENARIOS[n]['color'] for n in names]

        bars = ax_bar.bar(names, values, color=colors, alpha=0.8, width=0.6)
        
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f} s', ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax_bar.set_title('Średni Czas Przejazdu Pojazdów Uprzywilejowanych', fontsize=14, fontweight='bold')
        ax_bar.set_ylabel('Czas [s]', fontsize=12)
        ax_bar.grid(axis='y', alpha=0.3)
    else:
        ax_bar.text(0.5, 0.5, "Brak danych o pojazdach uprzywilejowanych", ha='center', fontsize=14)

    plt.tight_layout()
    plt.savefig('WYNIKI_POROWNANIE_FINAL.png', dpi=300)
    print("\n✅ Wykres zapisano jako: WYNIKI_POROWNANIE_FINAL.png")
    
    if stats_summary:
        print("\n=== PODSUMOWANIE LICZBOWE ===")
        print(f"{'Scenariusz':<30} | {'Śr. Kolejka':<12} | {'Wait Time':<10} | {'Czas Służb':<10}")
        print("-" * 70)
        for s in stats_summary:
            print(f"{s['Scenariusz']:<30} | {s['Śr. Kolejka']:<12.2f} | {s['Śr. Czas Oczekiwania']:<10.2f} | {s['Śr. Czas Służb']:<10.2f}")
    else:
        print("\n⚠️ Nie udało się wczytać żadnych danych. Sprawdź ścieżki do plików!")

if __name__ == "__main__":
    plot_results()