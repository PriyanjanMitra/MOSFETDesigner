import threading
from tkinter import messagebox, filedialog

import customtkinter as ctk
from sklearn.model_selection import train_test_split

from mosfet import MOSFETDesigner, load_and_preprocess_data, plot_training_history


class MOSFETDesignerGUI:
    def __init__(self, root):
        self.progress_bar = None
        self.status_label = None
        self.plot_path_entry = None
        self.ss_entry = None
        self.id_entry = None
        self.clear_btn = None
        self.clear_btn = None
        self.predict_btn = None
        self.tox_entry = None
        self.results_text = None
        self.save_plot_var = None
        self.save_plot_var = None
        self.vtgm_entry = None
        self.vtgm_entry = None
        self.save_plot_checkbox = None
        self.batch_entry = None
        self.epochs_entry = None
        self.load_csv_btn = None
        self.train_btn = None
        self.generate_btn = None
        self.material_dropdown = None
        self.material_var = ctk.StringVar(value="silicon")
        self.root = root
        self.root.title("MOSFET Designer - GUI")
        self.root.geometry("1000x1300")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.designer = None
        self.training_in_progress = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.create_widgets()

    def create_widgets(self):

        main_frame = ctk.CTkScrollableFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        material_frame = ctk.CTkFrame(main_frame)
        material_frame.pack(padx=20, pady=10, fill="x")

        material_label = ctk.CTkLabel(material_frame, text="Material:", font=("Arial", 12, "bold"))
        material_label.pack(side="left", padx=5)

        material_options = ["silicon", "sige", "gaas", "gan", "sic", "inp", "diamond"]
        self.material_dropdown = ctk.CTkComboBox(
            material_frame, values=material_options, variable=self.material_var,
            command=self.on_material_change
        )
        self.material_dropdown.pack(side="left", padx=5)

        training_frame = ctk.CTkFrame(main_frame)
        training_frame.pack(padx=20, pady=15, fill="x")

        training_label = ctk.CTkLabel(training_frame, text="Model Training", font=("Arial", 14, "bold"))
        training_label.pack(pady=10)

        button_frame = ctk.CTkFrame(training_frame)
        button_frame.pack(fill="x", pady=5)

        self.generate_btn = ctk.CTkButton(
            button_frame, text="Generate 1M Samples",
            command=self.generate_data_thread, width=150
        )
        self.generate_btn.pack(side="left", padx=5, pady=5)

        self.train_btn = ctk.CTkButton(
            button_frame, text="Train Model",
            command=self.train_model_thread, width=150
        )
        self.train_btn.pack(side="left", padx=5, pady=5)

        self.load_csv_btn = ctk.CTkButton(
            button_frame, text="Load CSV",
            command=self.load_csv, width=150
        )
        self.load_csv_btn.pack(side="left", padx=5, pady=5)

        train_options_frame = ctk.CTkFrame(training_frame)
        train_options_frame.pack(fill="x", pady=10)

        epochs_label = ctk.CTkLabel(train_options_frame, text="Epochs:", font=("Arial", 10))
        epochs_label.pack(side="left", padx=5)
        self.epochs_entry = ctk.CTkEntry(train_options_frame, placeholder_text="50", width=80)
        self.epochs_entry.pack(side="left", padx=5)
        self.epochs_entry.insert(0, "50")

        batch_label = ctk.CTkLabel(train_options_frame, text="Batch Size:", font=("Arial", 10))
        batch_label.pack(side="left", padx=5)
        self.batch_entry = ctk.CTkEntry(train_options_frame, placeholder_text="512", width=80)
        self.batch_entry.pack(side="left", padx=5)
        self.batch_entry.insert(0, "512")

        plot_frame = ctk.CTkFrame(training_frame)
        plot_frame.pack(fill="x", pady=10)

        self.save_plot_var = ctk.BooleanVar(value=False)
        self.save_plot_checkbox = ctk.CTkCheckBox(
            plot_frame, text="Save Training Plot",
            variable=self.save_plot_var
        )
        self.save_plot_checkbox.pack(side="left", padx=5)

        self.plot_path_entry = ctk.CTkEntry(plot_frame, placeholder_text="training_history.png")
        self.plot_path_entry.pack(side="left", padx=5, fill="x", expand=True)
        self.plot_path_entry.insert(0, "training_history.png")

        self.status_label = ctk.CTkLabel(training_frame, text="Status: Ready", text_color="green", font=("Arial", 10))
        self.status_label.pack(pady=5)

        self.progress_bar = ctk.CTkProgressBar(training_frame, mode='indeterminate')
        self.progress_bar.pack(fill="x", padx=5, pady=5)

        params_frame = ctk.CTkFrame(main_frame)
        params_frame.pack(padx=20, pady=15, fill="x")

        params_label = ctk.CTkLabel(params_frame, text="MOSFET Performance Parameters", font=("Arial", 14, "bold"))
        params_label.pack(pady=10)

        id_frame = ctk.CTkFrame(params_frame)
        id_frame.pack(fill="x", pady=8)
        ctk.CTkLabel(id_frame, text="Drain Current (Id) [A]:", width=150).pack(side="left", padx=5)
        self.id_entry = ctk.CTkEntry(id_frame, placeholder_text="1e-6")
        self.id_entry.pack(side="left", padx=5, fill="x", expand=True)

        ss_frame = ctk.CTkFrame(params_frame)
        ss_frame.pack(fill="x", pady=8)
        ctk.CTkLabel(ss_frame, text="Subthreshold Swing (SS) [mV/dec]:", width=150).pack(side="left", padx=5)
        self.ss_entry = ctk.CTkEntry(ss_frame, placeholder_text="80")
        self.ss_entry.pack(side="left", padx=5, fill="x", expand=True)

        vtgm_frame = ctk.CTkFrame(params_frame)
        vtgm_frame.pack(fill="x", pady=8)
        ctk.CTkLabel(vtgm_frame, text="Threshold Voltage (Vtgm) [V]:", width=150).pack(side="left", padx=5)
        self.vtgm_entry = ctk.CTkEntry(vtgm_frame, placeholder_text="0.3")
        self.vtgm_entry.pack(side="left", padx=5, fill="x", expand=True)

        tox_frame = ctk.CTkFrame(params_frame)
        tox_frame.pack(fill="x", pady=8)
        ctk.CTkLabel(tox_frame, text="Oxide Thickness (tox) [nm]:", width=150).pack(side="left", padx=5)
        self.tox_entry = ctk.CTkEntry(tox_frame, placeholder_text="1.0")
        self.tox_entry.pack(side="left", padx=5, fill="x", expand=True)

        prediction_frame = ctk.CTkFrame(main_frame)
        prediction_frame.pack(padx=20, pady=15, fill="x")

        pred_button_frame = ctk.CTkFrame(prediction_frame)
        pred_button_frame.pack(fill="x", pady=10)

        self.predict_btn = ctk.CTkButton(
            pred_button_frame, text="Predict Design",
            command=self.predict_design, width=150
        )
        self.predict_btn.pack(side="left", padx=5)

        self.clear_btn = ctk.CTkButton(
            pred_button_frame, text="Clear",
            command=self.clear_inputs, width=150
        )
        self.clear_btn.pack(side="left", padx=5)

        results_frame = ctk.CTkFrame(main_frame)
        results_frame.pack(padx=20, pady=15, fill="both", expand=True)

        results_label = ctk.CTkLabel(results_frame, text="Design Results & Status", font=("Arial", 14, "bold"))
        results_label.pack(pady=10)

        self.results_text = ctk.CTkTextbox(results_frame, height=300)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)

    def on_material_change(self, choice):
        try:
            self.designer = MOSFETDesigner(material=choice)
            self.status_label.configure(text=f"Status: Material changed to {choice}", text_color="blue")
        except Exception as e:
            self.status_label.configure(text=f"Status: Error - {str(e)}", text_color="red")

    def generate_data_thread(self):
        if self.training_in_progress:
            messagebox.showwarning("In Progress", "Training already in progress!")
            return

        thread = threading.Thread(target=self.generate_data)
        thread.daemon = True
        thread.start()

    def generate_data(self):
        try:
            self.training_in_progress = True
            self.progress_bar.start()
            self.status_label.configure(text="Status: Generating 1,000,000 samples...", text_color="orange")
            self.root.update()

            material = self.material_var.get()
            self.designer = MOSFETDesigner(material=material)

            X_train, y_train = self.designer.generate_synthetic_training_data(1000000)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

            self.progress_bar.stop()
            self.status_label.configure(text="Status: Data generated successfully!", text_color="green")
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end",
                                     f"✓ Generated 1,000,000 synthetic samples\n"
                                     f"✓ Training set: {len(self.X_train):,} samples\n"
                                     f"✓ Test set: {len(self.X_test):,} samples\n"
                                     f"✓ Material: {material}\n\n"
                                     f"Dataset Statistics:\n"
                                     f"  - Total features per sample: 4\n"
                                     f"  - Total output parameters per sample: 4\n"
                                     f"  - Memory usage: ~{(self.X_train.nbytes + self.y_train.nbytes) / (1024 ** 3):.2f} GB\n\n"
                                     f"Ready to train the model!"
                                     )

        except Exception as e:
            self.progress_bar.stop()
            self.status_label.configure(text=f"Status: Error - {str(e)}", text_color="red")
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")
        finally:
            self.training_in_progress = False

    def train_model_thread(self):
        if self.training_in_progress:
            messagebox.showwarning("In Progress", "Training already in progress!")
            return

        if self.X_train is None:
            messagebox.showerror("Error", "Please generate or load data first!")
            return

        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()

    def train_model(self):
        try:
            self.training_in_progress = True
            self.progress_bar.start()
            self.status_label.configure(text="Status: Training model...", text_color="orange")
            self.root.update()

            epochs = int(self.epochs_entry.get())
            batch_size = int(self.batch_entry.get())

            history = self.designer.train(
                self.X_train, self.y_train,
                X_val=self.X_test, y_val=self.y_test,
                epochs=epochs, batch_size=batch_size, verbose=0
            )

            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]

            if self.save_plot_var.get():
                plot_path = self.plot_path_entry.get()
                if not plot_path:
                    plot_path = "training_history.png"
                plot_training_history(history, plot_path)

            self.progress_bar.stop()
            self.status_label.configure(text="Status: Training complete!", text_color="green")
            self.results_text.delete("1.0", "end")

            results_str = f"✓ Model trained successfully!\n\n"
            results_str += f"Training Configuration:\n"
            results_str += f"  - Samples: 1,000,000\n"
            results_str += f"  - Training samples: {len(self.X_train):,}\n"
            results_str += f"  - Test samples: {len(self.X_test):,}\n"
            results_str += f"  - Epochs: {epochs}\n"
            results_str += f"  - Batch Size: {batch_size}\n\n"
            results_str += f"Final Metrics:\n"
            results_str += f"  - Training Loss: {final_loss:.6f}\n"
            results_str += f"  - Validation Loss: {final_val_loss:.6f}\n\n"

            if self.save_plot_var.get():
                results_str += f"✓ Training plot saved to: {self.plot_path_entry.get()}\n\n"

            results_str += f"The model is ready for predictions."

            self.results_text.insert("end", results_str)

        except ValueError:
            self.progress_bar.stop()
            self.status_label.configure(text="Status: Error - Invalid input", text_color="red")
            messagebox.showerror("Error", "Please enter valid numbers for epochs and batch size!")
        except Exception as e:
            self.progress_bar.stop()
            self.status_label.configure(text=f"Status: Error - {str(e)}", text_color="red")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
        finally:
            self.training_in_progress = False

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.progress_bar.start()
                self.status_label.configure(text="Status: Loading CSV...", text_color="orange")
                self.root.update()

                material = self.material_var.get()
                self.X_train, self.X_test, self.y_train, self.y_test = load_and_preprocess_data(file_path, material)
                self.designer = MOSFETDesigner(material=material)

                self.progress_bar.stop()
                self.status_label.configure(text="Status: CSV loaded successfully!", text_color="green")
                self.results_text.delete("1.0", "end")
                self.results_text.insert("end",
                                         f"✓ CSV loaded successfully!\n"
                                         f"✓ File: {file_path}\n"
                                         f"✓ Training set: {len(self.X_train):,} samples\n"
                                         f"✓ Test set: {len(self.X_test):,} samples\n\n"
                                         f"Ready to train the model!"
                                         )

            except Exception as e:
                self.progress_bar.stop()
                self.status_label.configure(text=f"Status: Error - {str(e)}", text_color="red")
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")

    def predict_design(self):
        try:
            if self.designer is None or self.designer.model is None:
                messagebox.showerror("Error", "Please train the model first!")
                return

            id_val = float(self.id_entry.get())
            ss_val = float(self.ss_entry.get())
            vtgm_val = float(self.vtgm_entry.get())
            tox_val = float(self.tox_entry.get())

            if id_val <= 0 or ss_val <= 0 or tox_val <= 0:
                messagebox.showerror("Error", "All values must be positive!")
                return

            results = self.designer.predict_design(id_val, ss_val, vtgm_val, tox_val)

            self.results_text.delete("1.0", "end")
            result_str = "NANOSCALE MOSFET DESIGN RESULTS\n" + "=" * 50 + "\n\n"
            for param, value in results.items():
                result_str += f"{param:30}: {value}\n"
            result_str += "\n" + "=" * 50
            result_str += "\nQuantum-aware design constraints applied"

            self.results_text.insert("end", result_str)
            self.status_label.configure(text="Status: Prediction successful!", text_color="green")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all parameters!")
            self.status_label.configure(text="Status: Error - Invalid input", text_color="red")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_label.configure(text=f"Status: Error - {str(e)}", text_color="red")

    def clear_inputs(self):
        self.id_entry.delete(0, "end")
        self.ss_entry.delete(0, "end")
        self.vtgm_entry.delete(0, "end")
        self.tox_entry.delete(0, "end")
        self.results_text.delete("1.0", "end")
        self.status_label.configure(text="Status: Ready", text_color="green")


if __name__ == "__main__":
    root = ctk.CTk()
    app = MOSFETDesignerGUI(root)
    root.mainloop()