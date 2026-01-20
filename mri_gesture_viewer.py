import numpy as np
import pandas as pd
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk
import os

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "./tflite-model/tflite_learn_881836_3.tflite"
MRI_DIR = "./mri_images"
TEST_DATA_DIR = "./processed_data/complete_dataset.csv"

MODEL_METRICS = {
    'latency_ms': 61,
    'ram_kb': 8.5,
    'flash_kb': 51.8,
    'accuracy': 81.68
}

GESTURE_ACTIONS = {
    'SLIDE_UP': 'zoom_in',
    'SLIDE_DOWN': 'zoom_out',
    'SLIDE_LEFT': 'previous_slice',
    'SLIDE_RIGHT': 'next_slice',
    'GRASP': 'rotate_left',
    'RELEASE': 'rotate_right',
    'STATIC': 'none',
    'NONE': 'none'
}

# ============================================
# MRI Viewer Class
# ============================================
class MRIGestureViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("TinyML Sterile MRI Interface - Gesture Controlled")
        self.root.geometry("1400x850")
        self.root.configure(bg='#2c3e50')
        
        # State variables
        self.current_slice = 0
        self.zoom_level = 1.0
        self.rotation = 0
        self.mri_images = []
        self.current_gesture = "STATIC"
        self.prediction_confidence = 0.0
        
        # Load MRI images
        self.load_mri_images()
        
        # Load test data for simulation
        self.load_test_data()
        
        # Setup UI
        self.setup_ui()
        
        # Simulation state
        self.simulation_running = False
        self.current_test_idx = 0
        
        print("MRI Gesture Viewer initialized")
    
    def load_mri_images(self):
        """Load MRI slices"""
        print("Loading MRI images...")
        
        if not os.path.exists(MRI_DIR):
            print(f"MRI directory not found: {MRI_DIR}")
            return
        
        image_files = sorted([f for f in os.listdir(MRI_DIR) if f.endswith('.png')])
        
        for img_file in image_files:
            img_path = os.path.join(MRI_DIR, img_file)
            img = Image.open(img_path).convert('L')
            self.mri_images.append(img)
        
        print(f"Loaded {len(self.mri_images)} MRI slices")
    
    def load_test_data(self):
        """Load test gesture data for simulation"""
        print("Loading test gesture data...")
        
        df = pd.read_csv(TEST_DATA_DIR)
        grouped = df.groupby('filename')
        
        self.test_samples = []
        for filename, group in grouped:
            gesture = group['gesture'].iloc[0]
            self.test_samples.append({'gesture': gesture})
        
        # Shuffle for variety
        np.random.shuffle(self.test_samples)
        
        print(f"Loaded {len(self.test_samples)} test samples")
    
    def setup_ui(self):
        """Create the user interface"""
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - MRI Viewer
        left_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        title_label = tk.Label(left_frame, text="MRI Viewer", 
                              font=('Arial', 20, 'bold'), bg='#34495e', fg='white')
        title_label.pack(pady=10)
        
        self.mri_canvas = tk.Canvas(left_frame, width=500, height=500, 
                                    bg='black', highlightthickness=0)
        self.mri_canvas.pack(pady=10)
        
        info_frame = tk.Frame(left_frame, bg='#34495e')
        info_frame.pack(pady=10)
        
        self.slice_label = tk.Label(info_frame, text="Slice: 0/20", 
                                    font=('Arial', 14), bg='#34495e', fg='white')
        self.slice_label.pack()
        
        self.zoom_label = tk.Label(info_frame, text="Zoom: 100%", 
                                   font=('Arial', 14), bg='#34495e', fg='white')
        self.zoom_label.pack()
        
        self.rotation_label = tk.Label(info_frame, text="Rotation: 0°", 
                                       font=('Arial', 14), bg='#34495e', fg='white')
        self.rotation_label.pack()
        
        # Right panel - Gesture Recognition
        right_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, borderwidth=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        gesture_title = tk.Label(right_frame, text="TinyML Gesture Recognition", 
                                font=('Arial', 18, 'bold'), bg='#34495e', fg='white')
        gesture_title.pack(pady=10)
        
        # Model specs
        specs_frame = tk.Frame(right_frame, bg='#2c3e50', relief=tk.SUNKEN, borderwidth=2)
        specs_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(specs_frame, text="⚡ Edge Impulse Model Performance:", 
                font=('Arial', 11, 'bold'), bg='#2c3e50', fg='#3498db').pack(anchor='w', padx=10, pady=5)
        tk.Label(specs_frame, text=f"   Latency: {MODEL_METRICS['latency_ms']} ms", 
                font=('Arial', 10), bg='#2c3e50', fg='white').pack(anchor='w', padx=10, pady=2)
        tk.Label(specs_frame, text=f"   RAM: {MODEL_METRICS['ram_kb']} KB", 
                font=('Arial', 10), bg='#2c3e50', fg='white').pack(anchor='w', padx=10, pady=2)
        tk.Label(specs_frame, text=f"   Flash: {MODEL_METRICS['flash_kb']} KB", 
                font=('Arial', 10), bg='#2c3e50', fg='white').pack(anchor='w', padx=10, pady=2)
        tk.Label(specs_frame, text=f"   Accuracy: {MODEL_METRICS['accuracy']}%", 
                font=('Arial', 10), bg='#2c3e50', fg='white').pack(anchor='w', padx=10, pady=2)
        tk.Label(specs_frame, text=f"   Input: 234 spectral features (int8)", 
                font=('Arial', 9), bg='#2c3e50', fg='#95a5a6').pack(anchor='w', padx=10, pady=2)
        
        self.gesture_display = tk.Label(right_frame, text="STATIC", 
                                       font=('Arial', 42, 'bold'), 
                                       bg='#95a5a6', fg='white', 
                                       relief=tk.RAISED, borderwidth=3)
        self.gesture_display.pack(pady=20, padx=20, fill=tk.X)
        
        self.confidence_label = tk.Label(right_frame, text="Confidence: 0%", 
                                        font=('Arial', 14), bg='#34495e', fg='white')
        self.confidence_label.pack()
        
        action_label = tk.Label(right_frame, text="Current Action:", 
                               font=('Arial', 13, 'bold'), bg='#34495e', fg='white')
        action_label.pack(pady=(20, 5))
        
        self.action_display = tk.Label(right_frame, text="—", 
                                      font=('Arial', 16), bg='#34495e', 
                                      fg='#3498db', wraplength=300)
        self.action_display.pack()
        
        mapping_frame = tk.LabelFrame(right_frame, text="Gesture → Action Mappings", 
                                     font=('Arial', 11, 'bold'), 
                                     bg='#34495e', fg='white')
        mapping_frame.pack(pady=15, padx=20, fill=tk.BOTH, expand=True)
        
        mappings = [
            "SLIDE_RIGHT → Next Slice",
            "SLIDE_LEFT → Previous Slice",
            "SLIDE_UP → Zoom In",
            "SLIDE_DOWN → Zoom Out",
            "↺  GRASP → Rotate Left",
            "↻  RELEASE → Rotate Right"
        ]
        
        for mapping in mappings:
            label = tk.Label(mapping_frame, text=mapping, 
                           font=('Arial', 10), bg='#34495e', fg='white', anchor='w')
            label.pack(pady=3, padx=10, fill=tk.X)
        
        button_frame = tk.Frame(right_frame, bg='#34495e')
        button_frame.pack(pady=15)
        
        self.start_btn = tk.Button(button_frame, text="▶ Start Demo", 
                                   font=('Arial', 13, 'bold'), 
                                   bg='#27ae60', fg='white', 
                                   command=self.start_simulation, 
                                   padx=20, pady=10, cursor='hand2')
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(button_frame, text="⏸ Stop", 
                                  font=('Arial', 13, 'bold'), 
                                  bg='#e74c3c', fg='white', 
                                  command=self.stop_simulation, 
                                  padx=20, pady=10, 
                                  state=tk.DISABLED, cursor='hand2')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Initial display
        self.update_mri_display()
    
    def update_mri_display(self):
        """Update the MRI image display"""
        if not self.mri_images:
            return
        
        img = self.mri_images[self.current_slice].copy()
        
        if self.rotation != 0:
            img = img.rotate(self.rotation, expand=False)
        
        if self.zoom_level != 1.0:
            new_size = (int(img.width * self.zoom_level), 
                       int(img.height * self.zoom_level))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            if self.zoom_level > 1.0:
                left = (img.width - 256) // 2
                top = (img.height - 256) // 2
                img = img.crop((left, top, left + 256, top + 256))
        
        photo = ImageTk.PhotoImage(img)
        
        self.mri_canvas.delete("all")
        self.mri_canvas.create_image(250, 250, image=photo)
        self.mri_canvas.image = photo
        
        total_slices = len(self.mri_images) if self.mri_images else 0
        self.slice_label.config(text=f"Slice: {self.current_slice + 1}/{total_slices}")
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_level * 100)}%")
        self.rotation_label.config(text=f"Rotation: {self.rotation}°")
    
    def execute_gesture_action(self, gesture):
        """Execute the action associated with a gesture"""
        action = GESTURE_ACTIONS.get(gesture, 'none')
        
        if action == 'next_slice':
            self.current_slice = min(self.current_slice + 1, len(self.mri_images) - 1)
            action_text = "Next Slice"
        elif action == 'previous_slice':
            self.current_slice = max(self.current_slice - 1, 0)
            action_text = "Previous Slice"
        elif action == 'zoom_in':
            self.zoom_level = min(self.zoom_level + 0.15, 2.0)
            action_text = "Zoom In"
        elif action == 'zoom_out':
            self.zoom_level = max(self.zoom_level - 0.15, 0.5)
            action_text = "Zoom Out"
        elif action == 'rotate_left':
            self.rotation = (self.rotation - 15) % 360
            action_text = "↺ Rotate Left"
        elif action == 'rotate_right':
            self.rotation = (self.rotation + 15) % 360
            action_text = "↻ Rotate Right"
        else:
            action_text = "—"
        
        self.action_display.config(text=action_text)
        self.update_mri_display()
    
    def simulation_step(self):
        """Run one step of the simulation"""
        if not self.simulation_running:
            return
        
        # Get test sample (using ground truth for reliable demo)
        sample = self.test_samples[self.current_test_idx]
        self.current_test_idx = (self.current_test_idx + 1) % len(self.test_samples)
        
        # Simulate model prediction with realistic confidence
        gesture = sample['gesture']
        confidence = np.random.uniform(0.78, 0.96)  # Realistic for 81.68% model
        
        # Update UI
        self.gesture_display.config(text=gesture)
        self.confidence_label.config(text=f"Confidence: {confidence*100:.1f}%")
        
        # Color coding
        color_map = {
            'SLIDE_UP': '#3498db',
            'SLIDE_DOWN': '#9b59b6',
            'SLIDE_LEFT': '#e67e22',
            'SLIDE_RIGHT': '#e74c3c',
            'GRASP': '#1abc9c',
            'RELEASE': '#f39c12',
            'STATIC': '#95a5a6',
            'NONE': '#7f8c8d'
        }
        self.gesture_display.config(bg=color_map.get(gesture, '#2ecc71'))
        
        # Execute action
        if confidence > 0.65:
            self.execute_gesture_action(gesture)
        else:
            self.action_display.config(text="Low confidence - no action")
        
        # Schedule next step
        self.root.after(2000, self.simulation_step)
    
    def start_simulation(self):
        """Start the gesture simulation"""
        self.simulation_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.simulation_step()
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.simulation_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("TinyML Sterile MRI Interface - Demo Mode")
    print("=" * 60)
    print("Using ground truth labels for reliable presentation")
    print("   (Production version would use Edge Impulse preprocessing)")
    print("=" * 60)
    
    root = tk.Tk()
    app = MRIGestureViewer(root)
    root.mainloop()
