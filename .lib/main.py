import traceback
import tkinter as tk
from tkinter import ttk, messagebox

from app.spectator_client import get_active_game_from_riot_id
from app.feature_builder import build_features
from app.predictor import Predictor


class LoLWinPredictorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LoL Live Win Predictor")
        self.root.geometry("700x500")
        self.root.resizable(False, False)

        self.predictor = None
        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill="both", expand=True)

        title_label = ttk.Label(
            main_frame,
            text="LoL Live Win Predictor",
            font=("Segoe UI", 18, "bold")
        )
        title_label.pack(pady=(0, 15))

        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill="x", pady=(0, 10))

        riot_id_label = ttk.Label(input_frame, text="Riot ID:")
        riot_id_label.pack(anchor="w")

        self.riot_id_entry = ttk.Entry(input_frame, font=("Segoe UI", 11))
        self.riot_id_entry.pack(fill="x", pady=(5, 0))
        self.riot_id_entry.insert(0, "name#tag")

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 10))

        self.predict_button = ttk.Button(
            button_frame,
            text="Predikovat",
            command=self.run_prediction
        )
        self.predict_button.pack(side="left")

        self.clear_button = ttk.Button(
            button_frame,
            text="Vymazat",
            command=self.clear_output
        )
        self.clear_button.pack(side="left", padx=(10, 0))

        output_label = ttk.Label(main_frame, text="Výstup:")
        output_label.pack(anchor="w")

        self.output_text = tk.Text(
            main_frame,
            height=20,
            wrap="word",
            font=("Consolas", 10)
        )
        self.output_text.pack(fill="both", expand=True)

        self._write_output("Aplikace připravena.\nZadej Riot ID a klikni na 'Predikovat'.")

    def _write_output(self, text: str):
        self.output_text.insert("end", text + "\n")
        self.output_text.see("end")

    def clear_output(self):
        self.output_text.delete("1.0", "end")

    def run_prediction(self):
        riot_id = self.riot_id_entry.get().strip()

        self.clear_output()

        if "#" not in riot_id:
            messagebox.showerror("Chyba", "Riot ID musí být ve formátu name#tag")
            return

        game_name, tag_line = riot_id.split("#", 1)

        try:
            self.predict_button.config(state="disabled")
            self.root.update_idletasks()

            self._write_output("Načítám účet...")
            game = get_active_game_from_riot_id(game_name, tag_line)

            if not game:
                self._write_output("Hráč není v aktivní hře nebo se nepodařilo načíst spectator data.")
                return

            self._write_output("Hra nalezena, vytvářím feature...")
            features = build_features(game)


            self._write_output("\nNačítám model...")
            if self.predictor is None:
                self.predictor = Predictor()

            self._write_output("Počítám predikci...")
            result = self.predictor.predict(features)

            blue = result["blue_win_prob"] * 100
            red = result["red_win_prob"] * 100

            self._write_output("\n=== VÝSLEDEK ===")
            self._write_output(f"Blue win chance: {blue:.2f}%")
            self._write_output(f"Red win chance:  {red:.2f}%")

        except Exception as e:
            self._write_output("\n=== CHYBA ===")
            self._write_output(str(e))
            self._write_output("\n=== TRACEBACK ===")
            self._write_output(traceback.format_exc())
            messagebox.showerror("Chyba", f"Nastala chyba:\n{e}")
        finally:
            self.predict_button.config(state="normal")


def main():
    root = tk.Tk()
    app = LoLWinPredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()