import io
import os
import re
import sys
import queue
import base64
import asyncio
import threading
import time
from datetime import date, datetime
from io import BytesIO

import flet as ft
import matplotlib.pyplot as plt
import numpy as np
import qrcode
import sounddevice as sd
import soundfile as sf
from flet import icons
from flet_contrib.color_picker import ColorPicker
from PIL import Image


def get_audio_devices():
    return sd.query_devices()

def record_audio(fs=44100, device=None):
    print(f"Recording... Device: {device}, Sample rate: {fs}")
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())
    try:
        stream = sd.InputStream(samplerate=fs, device=device, channels=1, callback=callback, dtype='float32')
        with stream:
            while True:
                yield q.get()
    except Exception as e:
        print(f"Error in record_audio: {e}")
        yield np.zeros((1024, 1))  # Return silent audio if there's an error


def save_audio(audio_data, filename="recorded_audio.wav", fs=44100):
    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)
    sf.write(filename, audio_data, fs, subtype='FLOAT')
    print(f"Audio saved to {filename}")

def generate_waveform(audio_data, fs=44100):
    plt.switch_backend('agg')
    plt.figure(figsize=(5, 1), facecolor='none', edgecolor='none')  # Reduced size
    plt.plot(np.linspace(0, len(audio_data) / fs, num=len(audio_data)), audio_data, color='#FFA500')  # Orange color
    plt.axis('off')
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, transparent=True)
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

class VoiceNote:
    def __init__(self, audio_data, fs):
        self.audio_data = audio_data
        self.fs = fs
        self.is_playing = False
        self.playback_position = 0
        self.duration = len(audio_data) / fs
        self.current_time = 0
        self.start_time = 0

class VerticalProgressBar(ft.UserControl):
    def __init__(self, value, height=100, color="green", bgcolor="#EEEEEE"):
        super().__init__()
        self.value = value
        self.height = height
        self.color = color
        self.bgcolor = bgcolor

    def build(self):
        return ft.Container(
            width=10,  # Adjust width as needed
            height=self.height,
            bgcolor=self.bgcolor,
            border_radius=ft.border_radius.all(5),
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            content=ft.Container(
                height=self.height * self.value,
                bgcolor=self.color,
                alignment=ft.alignment.bottom_center,  # Align the progress from the bottom
            ),
        )

class VoiceTask(ft.UserControl):
    def __init__(self, page, task_name, task_delete, task_status_change, parent_container, handle_dismissal):
        super().__init__()
        self.locked = False
        self.page = page
        self.full_task_name = task_name
        self.task_name = self.format_task_name(task_name)
        self.task_delete = task_delete
        self.task_status_change = task_status_change
        self.parent_container = parent_container
        self.handle_dismissal = handle_dismissal  # Store the handle_dismissal function
        self.description_field = None
        
        self.play_pause_buttons = {} 
        # self.init_ui_elements()
        # Initialize buttons
        self.priority_colors = {
            "No priority": ft.colors.BLACK,
            "Highest": ft.colors.RED,
            "High": ft.colors.PURPLE,
            "Medium": ft.colors.ORANGE,
            "Low": ft.colors.BLUE,
            "Lowest": ft.colors.GREEN,
        }
        self.live_waveform = ft.Image(visible=False)
        self.live_waveform_container = ft.Container(
            content=self.live_waveform,
            visible=False,
            width=200,
            height=50,
        )
        self.current_priority = "No priority"
        self.search_descriptions = ft.TextField(
            # height=30,
            hint_text="find descriptions...",
            expand=True,
            on_change=self.filter_descriptions,
            prefix_icon=ft.icons.SEARCH,
            bgcolor=ft.colors.BLUE_50,
            border_radius=ft.border_radius.all(25),
            border=ft.border.all(1, ft.colors.GREY_400),
            text_style=ft.TextStyle(color=ft.colors.WHITE),
            hint_style=ft.TextStyle(color=ft.colors.GREY_400),
            color=ft.colors.BLACK,
            autofocus=True,
            autocorrect=True,
            enable_suggestions=True,
            capitalization=ft.TextCapitalization.SENTENCES,
            cursor_width=2,
            cursor_color=ft.colors.BLUE_600,
            focused_border_color=ft.colors.BLUE_400,
            focused_bgcolor=ft.colors.BLUE_100,
            suffix_icon=ft.icons.CLEAR,
            tooltip="Enter search terms",
            keyboard_type=ft.KeyboardType.TEXT,
            selection_color=ft.colors.BLUE_200,
            # max_length=50,
            # counter_text="0/50",
            # helper_text="Search in task descriptions",
            error_style=ft.TextStyle(color=ft.colors.RED_400),
            dense=True
        )
                
        self.play_pause_button = ft.IconButton(
            icon=ft.icons.PLAY_ARROW,
            tooltip="Play Voice Note",
            on_click=self.toggle_playback,
            visible=False,
        )
        self.pause_button = ft.IconButton(
            icon=ft.icons.PAUSE,
            tooltip="Pause Voice Note",
            on_click=self.pause_playback,
            visible=False,
        )
        self.resume_button = ft.IconButton(
            icon=ft.icons.PLAY_CIRCLE_FILLED,
            tooltip="Resume Voice Note",
            on_click=self.resume_playback,
            visible=False,
        )
        # self.volume_bar = VerticalProgressBar(value=0, height=100, color="green", bgcolor="#EEEEEE")
        self.volume_bar = ft.ProgressBar(
            expand=True,  # Make it stretchable
            color=ft.colors.ORANGE,  # Changed to orange for consistency with instruction
            bgcolor=ft.colors.GREY_300,  # Adding a background color for contrast
            value=0.5,  # Set an initial value for demonstration
            tooltip="Volume Level",  # Adding a tooltip for user guidance
            width=None,  # Set to None to allow it to stretch
            height=5,  # Fixed height
        )
        
        # self.volume_bar = ft.ProgressBar(
        #     width=100, 
        #     color="green", 
        #     visible=False
        #     )
        self.record_button = ft.IconButton(
            icon=ft.icons.MIC,
            tooltip="Record Voice Note",
            on_click=self.toggle_recording,
            icon_color=ft.colors.RED,
            icon_size=20,
        )
        self.play_pause_button = ft.IconButton(
            icon=ft.icons.PLAY_ARROW,
            tooltip="Play Voice Note",
            on_click=self.toggle_playback,
            visible=False,
        )
        
        #////////////////////////////////////////////////////////////////////////
        #////////////////////////////////////////////////////////////////////////
        self.display_task = ft.Checkbox(
            value=False,
            label=self.task_name,
            on_change=self.status_changed
        )
        self.display_task.label_style = ft.TextStyle(weight=ft.FontWeight.BOLD, size=14, color=ft.colors.BLACK)
        
        self.image_button = ft.IconButton(
            icon=ft.icons.IMAGE,
            tooltip="Add Image or Background Color",
            on_click=self.show_image_dialog,
        )

        self.priority_dropdown = ft.PopupMenuButton(
            icon=ft.icons.FLAG,
            tooltip="Set Priority",
            items=[
                ft.PopupMenuItem(text=priority, on_click=self.set_priority)
                for priority in self.priority_colors.keys()
            ]
        )
        self.edit_name = ft.TextField(expand=1)
        self.waveform = ft.Image(visible=False, height=50)
        self.description_preview = ft.Text("", visible=False)

        # Initialize buttons
        self.record_button = ft.IconButton(
            icon=ft.icons.MIC,
            tooltip="Record Voice Note",
            on_click=self.toggle_recording,
            icon_color=ft.colors.RED,
            icon_size=22,
        )
        self.play_pause_button = ft.IconButton(
            icon=ft.icons.PLAY_ARROW,
            tooltip="Play Voice Note",
            on_click=self.toggle_playback,
            visible=False,
        )
        self.description_button = ft.IconButton(
            icon=ft.icons.DESCRIPTION_OUTLINED,
            tooltip="Add Description",
            on_click=self.add_description_clicked,
        )
        self.expand_button = ft.IconButton(
            icon=ft.icons.EXPAND_MORE,
            on_click=self.toggle_expand
        )

        self.voice_notes = []
        self.voice_notes_container = ft.Column()
        
        #////////////////////////////////////////////////////////////////////////
        #////////////////////////////////////////////////////////////////////////
        self.expand_button = ft.Ref[ft.IconButton]()
        self.priority_dropdown = ft.Ref[ft.PopupMenuButton]()
    
        self.selection_start = 0
        self.selection_end = 0
        self.current_style = ft.TextStyle()
        self.expanded = False
        self.description = ""
        self.formatting = []
        self.due_date = None
        self.due_date_picker = None
        self.audio_data = None
        self.is_recording = False
        self.fs = 44100
        self.input_device = None
        self.record_thread = None
        self.record_generator = None
        self.is_playing = False
        self.audio_playback = None
        self.playback_position = 0
        self.alarm_active = False
        self.alarm_time = ft.Ref[ft.TimePicker]()
        self.alarm_time_text = ft.Text("Alarm not set", visible=False)

        self.qr_code_image = None
        self.qr_code_image = ft.Image(src_base64="", width=150, height=150)
        self.password_field = None
        self.confirm_password_field = None
        self.email_field = None
        self.secret_question_field = None
        self.secret_answer_field = None

        self.descriptions = []  # List to store multiple descriptions
        self.descriptions_container = ft.Column()  # Container to display descriptions
                
        self.detail_tab = self.build_detail_view()
        self.mind_map_tab = self.build_mind_map_view()
        self.description_tab = self.build_description_view()
        
        self.tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(text="Details", content=self.detail_tab),
                ft.Tab(text="Mind Map", content=self.mind_map_tab),
                ft.Tab(text="Description", content=self.description_tab),
            ],
            expand=1
        )
        self.expandable_view = ft.Container(
            content=self.tabs,
            visible=False
        )
        
        self.display_task = ft.Checkbox(
            value=False,
            label=self.task_name,
            on_change=self.status_changed
        )
        self.display_task.label_style = ft.TextStyle(weight=ft.FontWeight.BOLD)  # Make task name bold
        self.display_task.label_style = ft.TextStyle(size=14)  # Make task name bold
        self.display_task.label_style = ft.TextStyle(color=ft.colors.BLACK)  # Make task name bold
        
        self.image_button = ft.IconButton(
            icon=ft.icons.IMAGE,
            tooltip="Add Image or Background Color",
            on_click=self.show_image_dialog,
        )

        self.priority_dropdown = ft.PopupMenuButton(
            icon=ft.icons.FLAG,
            tooltip="Set Priority",
            items=[
                ft.PopupMenuItem(text=priority, on_click=self.set_priority)
                for priority in self.priority_colors.keys()
            ]
        )
        self.edit_name = ft.TextField(expand=1)
        self.waveform = ft.Image(visible=False, height=40)
        self.description_preview = ft.Text("", visible=False)

        self.play_pause_button = ft.IconButton(
            icon=ft.icons.PLAY_ARROW,
            tooltip="Play Voice Note",
            on_click=self.toggle_playback,
            visible=False,
        )
        self.description_button = ft.IconButton(
            icon=ft.icons.DESCRIPTION_OUTLINED,
            tooltip="Add Description",
            on_click=self.add_description_clicked,
        )

        self.voice_notes = []
        self.voice_notes_container = ft.Column()

        # Build views
        self.display_view = self.build_display_view()
        self.edit_view = self.build_edit_view()
        self.detail_view = self.build_detail_view()
        self.detail_view.visible = False

        # Create drop container
        self.drop_container = ft.Container(
            content=ft.Column([
                self.display_view,
                self.edit_view,
                self.detail_view
            ]),
            padding=3,  # Increased padding for better spacing
            border=ft.border.all(2, ft.colors.BLUE_400),  # Thicker border with a nice blue color
            border_radius=ft.border_radius.all(10),  # Increased border radius for smoother corners
            expand=True,
            gradient=ft.LinearGradient(
                begin=ft.alignment.top_left,
                end=ft.alignment.bottom_right,
                colors=[ft.colors.BLUE_50, ft.colors.INDIGO_50]
            ),  # Add a subtle gradient background
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=10,
                color=ft.colors.BLUE_GREY_300,
                offset=ft.Offset(0, 5),
            ),  # Add a soft shadow for depth
        )

        
        self.update_task_color()

    def build(self):
        return self.drop_container
    
    def generate_live_waveform(self, audio_chunk):
        plt.figure(figsize=(4, 1), facecolor='none', edgecolor='none')
        plt.plot(audio_chunk, color='#FFA500')  # Orange color
        plt.axis('off')
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, transparent=True)
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def matches_search(self, search_term):
        if search_term.lower() in self.full_task_name.lower():
            return True
        for description in self.descriptions:
            if search_term.lower() in description.lower():
                return True
        return False
    
    def format_task_name(self, name):
        if len(name) > 18:
            return name[:17] + "…"  # Use ellipsis character
        return name.ljust(18)  # Pad with spaces if shorter than 18 characters

    def build_mind_map_view(self):
        # Placeholder for mind map creation area
        return ft.Container(
            content=ft.Column([
                ft.Text("Mind Map Creation Area", size=20, weight=ft.FontWeight.BOLD),
                ft.ElevatedButton("Mind Map", on_click=self.create_mind_map),
                # Add more controls for mind map creation here
            ]),
            padding=5,
            border_radius=ft.border_radius.all(5),
        )

    def build_description_view(self):
        self.description_field = ft.TextField(
            multiline=True,
            min_lines=3,
            max_lines=90,
            value="",
            label="Task Description",
            on_change=self.update_description
        )
        return ft.Container(
            content=ft.Column([
                self.description_field,
                ft.ElevatedButton("Save Description", on_click=self.save_description_and_close),
                self.descriptions_container  # This should be here, not added to the page
            ]),
            padding=5,
            border_radius=ft.border_radius.all(5),
        )
    
    
    def build_display_view(self):
        self.more_options_menu = self.create_more_options_menu()
        self.lock_button = ft.IconButton(
            icon=ft.icons.LOCK_OPEN,
            tooltip="Lock/Unlock Task",
            on_click=self.toggle_lock
        )
        self.edit_button = ft.IconButton(
            icon=ft.icons.EDIT,
            tooltip="Rename",
            on_click=self.edit_clicked
        )
        # self.fingerprint_button = ft.IconButton(
        #     icon=ft.icons.FINGERPRINT,
        #     tooltip="Biometric Lock",
        #     # on_click=self.biometric_lock
        # )

        self.share_button = ft.IconButton(
            icon=ft.icons.SHARE,
            on_click=self.share_task,
            icon_color=ft.colors.TEAL_400,  # Unique teal color
            tooltip="Share Task",
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                overlay_color=ft.colors.TEAL_100,
            ),
        )
        self.qrcode_button = ft.IconButton(
            icon=ft.icons.QR_CODE,
            on_click=self.generate_qr_code,
            icon_color=ft.colors.INDIGO_400,  # Unique indigo color
            tooltip="Generate QR Code",
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                overlay_color=ft.colors.INDIGO_100,
            ),
        )
        
        return ft.Column([
        
            ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Row([
                        self.display_task,
                        ft.Tooltip(
                            message=self.full_task_name,
                            content=ft.Icon(ft.icons.INFO),
                            visible=len(self.full_task_name) > 18
                        ),
                        ft.PopupMenuButton(
                            ref=self.priority_dropdown,
                            icon=ft.icons.FLAG,
                            tooltip="Set Priority",
                            items=[
                                ft.PopupMenuItem(text=priority, on_click=self.set_priority)
                                for priority in self.priority_colors.keys()
                            ]
                        ),
                    ]),
                    ft.Row([
                        self.lock_button,
                        self.edit_button,
                        # self.fingerprint_button,
                        self.share_button,
                        self.qrcode_button,
                                                
                        ft.IconButton(
                            ref=self.expand_button,
                            icon=ft.icons.EXPAND_MORE,
                            on_click=self.toggle_expand
                        ),
                        self.more_options_menu,
                    ]),
                ]),
            
            ft.Row([
                 ft.Container(  # Special row for volume bar
            content=self.volume_bar,
            # height=10,  # Fixed height
            width=None,  # Set to None to allow it to stretch
            expand=True,  # Make it stretchable
            # padding=ft.padding.only(left=10, right=10),
            alignment=ft.alignment.center,  # Center the volume bar
            # visible=False,  # Initially invisible
        ),
                
                # # self.fingerprint_button,
                # self.share_button,
                # self.qrcode_button,
                                        
                # ft.IconButton(
                #     ref=self.expand_button,
                #     icon=ft.icons.EXPAND_MORE,
                #     on_click=self.toggle_expand
                # ),
                # self.more_options_menu,
            ]),
            
        
        ])

    def filter_descriptions(self, e):
        search_term = self.search_descriptions.value.lower()
        for index, desc_container in enumerate(self.descriptions_container.controls):
            desc_text = desc_container.content.controls[0]
            if search_term in self.descriptions[index].lower():
                desc_container.visible = True
            else:
                desc_container.visible = False
        self.update()
    
    def format_task_name(self, name):
        if len(name) > 18:
            return name[:17] + "…"  # Use ellipsis character
        return name.ljust(18)  # Pad with spaces if shorter than 18 characters

    def edit_clicked(self, e):
        self.edit_name.value = self.full_task_name
        self.display_view.visible = False
        self.edit_view.visible = True
        self.update()
    
    def edit_description(self, index):
        if 0 <= index < len(self.descriptions):
            self.description_field.value = self.descriptions[index]
            self.description_field.data = index  # Store the index being edited
            self.show_description_dialog(existing_description=self.descriptions[index], edit_index=index)
        else:
            print(f"Invalid index: {index}")
        self.page.update()
    
    def delete_description(self, index):
        if 0 <= index < len(self.descriptions):
            self.descriptions.pop(index)
            self.update_descriptions_ui()
        else:
            print(f"Invalid index: {index}")
        
    def save_clicked(self, e):
        new_name = self.edit_name.value
        if new_name and new_name != self.full_task_name:
            self.full_task_name = new_name
            self.task_name = self.format_task_name(new_name)
            self.display_task.label = self.task_name
        self.display_view.visible = True
        self.edit_view.visible = False
        self.update()
    #-------------------------------------------------------
    def xxshow_lock_dialog(self):
        self.qr_code_image = ft.Image(width=200, height=200)

        def update_qr_code(e):
            if all([self.password_field, self.email_field, self.secret_question_field, self.secret_answer_field]):
                qr_data = self.generate_qr_code(
                    self.password_field.value,
                    self.email_field.value,
                    self.secret_question_field.value,
                    self.secret_answer_field.value
                )
                self.qr_code_image.src_base64 = qr_data
                self.qr_code_image.update()

        self.password_field = ft.TextField(
            label="Password",
            password=True,
            can_reveal_password=True,
            on_change=update_qr_code
        )
        self.confirm_password_field = ft.TextField(
            label="Confirm Password",
            password=True,
            can_reveal_password=True,
            on_change=update_qr_code
        )
        self.email_field = ft.TextField(
            label="Safe Email",
            on_change=update_qr_code
        )
        self.secret_question_field = ft.TextField(
            label="Secret Question",
            on_change=update_qr_code
        )
        self.secret_answer_field = ft.TextField(
            label="Secret Answer",
            password=True,
            can_reveal_password=True,
            on_change=update_qr_code
        )

        # self.page.bottom_appbar = ft.BottomAppBar(
        #     bgcolor=ft.colors.BLUE,
        #     shape=ft.NotchShape.CIRCULAR,
        #     content=ft.Row(
        #         controls=[
        #             ft.IconButton(icon=ft.icons.MENU, icon_color=ft.colors.WHITE),
        #             ft.Container(expand=True),
        #             ft.IconButton(icon=ft.icons.SEARCH, icon_color=ft.colors.WHITE),
        #             ft.IconButton(icon=ft.icons.FAVORITE, icon_color=ft.colors.WHITE),
        #         ]
        #     ),
        # )
        self.drawer = ft.NavigationDrawer(
        on_dismiss=self.handle_dismissal, # Pass handle_dismissal here
        # on_change=handle_change,
        controls=[
            ft.Container(height=12),

            ft.Divider(thickness=2),
            ft.NavigationDrawerDestination(
                icon_content=ft.Icon(ft.icons.MAIL_OUTLINED),
                label="Item 2",
                selected_icon=ft.icons.MAIL,
            ),
            ft.NavigationDrawerDestination(
                icon_content=ft.Icon(ft.icons.PHONE_OUTLINED),
                label="Item 3",
                selected_icon=ft.icons.PHONE,
            ),
        ],
    )

        self.page.add(ft.ElevatedButton("Show drawer", on_click=lambda e: self.page.open(drawer)))
    
        def confirm_lock(e):
            if self.password_field.value == self.confirm_password_field.value:
                self.lock_password = self.password_field.value
                self.safe_email = self.email_field.value
                self.secret_question = self.secret_question_field.value
                self.secret_answer = self.secret_answer_field.value
                self.qr_code_data = self.qr_code_image.src_base64
                dialog.open = False
                self.page.update()
                self.perform_lock()
            else:
                self.page.snack_bar = ft.SnackBar(content=ft.Text("Passwords do not match"))
                self.page.snack_bar.open = True
                self.page.update()

        dialog_content = ft.Column([
            self.qr_code_image,
            self.password_field,
            self.confirm_password_field,
            self.email_field,
            self.secret_question_field,
            self.secret_answer_field,
        ], tight=True, spacing=20, alignment=ft.MainAxisAlignment.CENTER)

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Lock Task"),
            content=dialog_content,
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: setattr(dialog, 'open', False)),
                ft.TextButton("Lock", on_click=confirm_lock),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

        # Generate initial QR code after the dialog is added to the page
        update_qr_code(None)

    def show_unlock_dialog(self):
        password_field = ft.TextField(
            label="Password",
            password=True,
            can_reveal_password=True,
        )

        def confirm_unlock(e):
            if password_field.value == self.lock_password:
                dialog.open = False
                self.page.update()
                self.perform_unlock()
            else:
                self.page.snack_bar = ft.SnackBar(content=ft.Text("Incorrect password"))
                self.page.snack_bar.open = True
                self.page.update()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Unlock Task"),
            content=ft.Column([
                password_field,
            ], tight=True, spacing=20, alignment=ft.MainAxisAlignment.CENTER),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: setattr(dialog, 'open', False)),
                ft.TextButton("Unlock", on_click=confirm_unlock),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.dialog = dialog
        dialog.open = True
        self.page.update()
        
    def generate_qr_code(self, data):
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def perform_lock(self):
        self.locked = True
        self.display_task.disabled = True
        self.display_task.label = f"🔒 {self.task_name}"
        self.expand_button.current.disabled = True
        self.priority_dropdown.current.disabled = True
        self.update()

    def perform_unlock(self):
        self.locked = False
        self.display_task.disabled = False
        self.display_task.label = self.task_name
        self.expand_button.current.disabled = False
        self.priority_dropdown.current.disabled = False
        self.update()
    
    
    def show_unlock_dialog(self, e):
        password_field = ft.TextField(
            label="Password",
            password=True,
            can_reveal_password=True,
        )

        def confirm_unlock(e):
            if password_field.value == self.lock_password:
                dialog.open = False
                self.page.update()
                self.perform_unlock()
            else:
                self.page.snack_bar = ft.SnackBar(content=ft.Text("Incorrect password"))
                self.page.snack_bar.open = True
                self.page.update()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Unlock Task"),
            content=ft.Column([
                password_field,
            ], tight=True, spacing=20, alignment=ft.MainAxisAlignment.CENTER),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: setattr(dialog, 'open', False)),
                ft.TextButton("Unlock", on_click=confirm_unlock),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.dialog = dialog
        dialog.open = True
        self.page.update()
        
    def perform_unlock(self):
        self.locked = False
        self.display_task.disabled = False
        self.display_task.label = self.task_name
        self.expand_button.current.disabled = False
        self.priority_dropdown.current.disabled = False
        self.update()

    def toggle_lock(self, e):
        if not self.locked:
            self.show_lock_dialog()
        else:
            self.show_unlock_dialog(e)

        # Update the lock button icon
        self.lock_button.icon = ft.icons.LOCK if self.locked else ft.icons.LOCK_OPEN
        self.update()

    def show_lock_dialog(self):
        def update_qr_code(e):
            qr_data = f"Password: {self.password_field.value}\nEmail: {self.email_field.value}\nQuestion: {self.secret_question_field.value}\nAnswer: {self.secret_answer_field.value}"
            qr_image = self.generate_qr_code(qr_data)
            if self.qr_code_image:
                self.qr_code_image.src_base64 = qr_image
                self.qr_code_image.update()
            else:
                print("QR code image not initialized")

        self.password_field = ft.TextField(
            label="Password",
            password=True,
            can_reveal_password=True,
            on_change=update_qr_code
        )
        self.confirm_password_field = ft.TextField(
            label="Confirm Password",
            password=True,
            can_reveal_password=True,
            on_change=update_qr_code
        )
        self.email_field = ft.TextField(
            label="Safe Email",
            on_change=update_qr_code
        )
        self.secret_question_field = ft.TextField(
            label="Secret Question",
            on_change=update_qr_code
        )
        self.secret_answer_field = ft.TextField(
            label="Secret Answer",
            password=True,
            can_reveal_password=True,
            on_change=update_qr_code
        )

        def confirm_lock(e):
            if self.password_field.value == self.confirm_password_field.value:
                self.lock_password = self.password_field.value
                self.safe_email = self.email_field.value
                self.secret_question = self.secret_question_field.value
                self.secret_answer = self.secret_answer_field.value
                self.qr_code_data = self.qr_code_image.src_base64
                dialog.open = False
                self.page.update()
                self.perform_lock()
            else:
                self.page.snack_bar = ft.SnackBar(content=ft.Text("Passwords do not match"))
                self.page.snack_bar.open = True
                self.page.update()

        dialog_content = ft.Column([
            self.qr_code_image if self.qr_code_image else ft.Text("QR Code placeholder"),
            self.password_field,
            self.confirm_password_field,
            self.email_field,
            self.secret_question_field,
            self.secret_answer_field,
        ], tight=True, spacing=20, alignment=ft.MainAxisAlignment.CENTER)

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Lock Task"),
            content=dialog_content,
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: setattr(dialog, 'open', False)),
                ft.TextButton("Lock", on_click=confirm_lock),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

        # Generate initial QR code
        update_qr_code(None)
            
    def toggle_expand(self, e):
        if not self.locked:
            self.expanded = not self.expanded
            self.expand_button.current.icon = ft.icons.EXPAND_LESS if self.expanded else ft.icons.EXPAND_MORE
            self.detail_view.visible = self.expanded
            self.update()
            
    def lock_task(self, e):
        self.locked = not self.locked
        if self.locked:
            self.display_task.disabled = True
            self.display_task.label = f"🔒 {self.task_name}"
        else:
            self.display_task.disabled = False
            self.display_task.label = self.task_name
        self.update()
        print(f"{'Locked' if self.locked else 'Unlocked'} task: {self.task_name}")

    def share_task(self, e):
        share_text = f"Check out my task: {self.task_name}"
        if self.description:
            share_text += f"\nDescription: {self.description}"
        if self.due_date:
            share_text += f"\nDue date: {self.due_date.strftime('%Y-%m-%d')}"

        def share_to_platform(platform):
            print(f"Sharing to {platform}: {share_text}")
            self.page.snack_bar = ft.SnackBar(content=ft.Text(f"Shared to {platform}"))
            self.page.snack_bar.open = True
            self.page.update()
            share_dialog.open = False            
            self.page.update()

        def close_share_dialog(e):
            share_dialog.open = False
            self.page.update()

        share_buttons = [
            ft.ElevatedButton(
                content=ft.Row([
                    ft.Icon(name=ft.icons.FACEBOOK),
                    ft.Text("Facebook")
                ]),
                on_click=lambda _: share_to_platform("Facebook")
            ),
            ft.ElevatedButton(
                content=ft.Row([
                    ft.Text("WhatsApp")
                ]),
                on_click=lambda _: share_to_platform("WhatsApp")
            ),
            ft.ElevatedButton(
                content=ft.Row([
                    ft.Icon(name=ft.icons.MAIL),
                    ft.Text("Email")
                ]),
                on_click=lambda _: share_to_platform("Email")
            ),
            ft.ElevatedButton(
                content=ft.Row([
                    ft.Icon(name=ft.icons.SEND),
                    ft.Text("Twitter")
                ]),
                on_click=lambda _: share_to_platform("Twitter")
            ),
            ft.ElevatedButton(
                content=ft.Row([
                    ft.Icon(name=ft.icons.WORK),
                    ft.Text("LinkedIn")
                ]),
                on_click=lambda _: share_to_platform("LinkedIn")
            ),
            ft.ElevatedButton(
                content=ft.Row([
                    ft.Icon(name=ft.icons.PHOTO_CAMERA),
                    ft.Text("Instagram")
                ]),
                on_click=lambda _: share_to_platform("Instagram")
            ),
            ft.ElevatedButton(
                content=ft.Row([
                    ft.Icon(name=ft.icons.TELEGRAM),
                    ft.Text("Telegram")
                ]),
                on_click=lambda _: share_to_platform("Telegram")
            ),
            ft.ElevatedButton(
                content=ft.Row([
                    ft.Icon(name=ft.icons.REDDIT),
                    ft.Text("Reddit")
                ]),
                on_click=lambda _: share_to_platform("Reddit")
            ),
            ft.ElevatedButton(
                content=ft.Row([
                    ft.Icon(name=ft.icons.CHAT),
                    ft.Text("SMS")
                ]),
                on_click=lambda _: share_to_platform("SMS")
            ),
            ft.ElevatedButton(
                content=ft.Row([
                    ft.Icon(name=ft.icons.LINK),
                    ft.Text("Copy Link")
                ]),
                on_click=lambda _: share_to_platform("Clipboard")
            ),
        ]

        share_dialog = ft.AlertDialog(
            title=ft.Text("Share Task"),
            content=ft.Column(
                [ft.Text("Choose a platform to share your task:")]
                + share_buttons,
                tight=True,
                spacing=10,
                run_spacing=10,
                scroll=ft.ScrollMode.AUTO,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=close_share_dialog),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.dialog = share_dialog
        share_dialog.open = True
        self.page.update()
    
    # def share_task(self, e):
    #     share_text = f"Check out my task: {self.task_name}"
    #     if self.description:
    #         share_text += f"\nDescription: {self.description}"
    #     if self.due_date:
    #         share_text += f"\nDue date: {self.due_date.strftime('%Y-%m-%d')}"
        
    #     # Here you would typically use a sharing API or clipboard functionality
    #     print(f"Sharing task: {share_text}")
    #     # For now, let's just show a snackbar with the share text
    #     self.page.snack_bar = ft.SnackBar(content=ft.Text(f"Sharing: {share_text}"))
    #     self.page.snack_bar.open = True
    #     self.page.update()

    def duplicate_task(self, e):
        new_task = VoiceTask(
            self.page,
            f"Copy of {self.task_name}",
            self.task_delete,
            self.task_status_change,
            self.parent_container,
            self.handle_dismissal  # Add this argument
        )
        # Copy relevant attributes from self to new_task
        new_task.description = self.description
        new_task.due_date = self.due_date
        new_task.current_priority = self.current_priority
        # ... (copy other relevant attributes) ...

        # Add the new task to the parent container
        self.parent_container.controls.append(new_task)
        self.parent_container.update()
        print(f"Duplicated task: {self.task_name}")
    #-------------------------------------------------------
    #-------------------------------------------------------
    def show_priority_dialog(self, e):
        print("Showing priority dialog")
        # Implement priority dialog functionality

    def add_reminder(self, e):
        print("Adding reminder")
        # Implement reminder functionality

    def add_attachment(self, e):
        print("Adding attachment")
        # Implement attachment functionality

    def add_tags(self, e):
        print("Adding tags")
        # Implement tag functionality

    def export_task(self, e):
        print("Exporting task")
        # Implement task export functionality

    def add_subtasks(self, e):
        print("Adding subtasks")
        # Implement subtask functionality

    def add_collaborator(self, e):
        print("Adding collaborator")
        # Implement collaborator functionality

    def add_location(self, e):
        print("Adding location")
        # Implement location functionality
    
    def add_notes(self, e):
        print("Adding notes")
        # Implement notes functionality

    def add_time_estimate(self, e):
        print("Adding time estimate")
        # Implement time estimate functionality

    def add_project(self, e):
        print("Adding project")
        # Implement project functionality

    def set_deadline(self, e):
        print("Setting deadline")
        # Implement deadline setting functionality

    def set_progress(self, e):
        print("Setting progress")
        # Implement progress setting functionality

    def create_mind_map(self, e):
        print("Creating mind map")
        # Implement mind map creation functionality

    def add_voice_command(self, e):
        print("Voice command")
        # Implement voice command functionality

    def create_time_lapse(self, e):
        print("Creating time-lapse")
        # Implement time-lapse creation functionality

    def generate_task_flowchart(self, e):
        print("Generating task flowchart")
        # Implement task flowchart generation functionality

    def generate_biometric_lock(self, e):
        print("Generating biometric lock")
        # Implement biometric lock functionality

    def link_task(self, e):
        print("Linking task")
        # Implement task linking functionality

    def create_subtask(self, e):
        print("Creating subtask")
        # Implement subtask creation functionality
    #-------------------------------------------------------
    #-------------------------------------------------------
    def build_edit_view(self):
        return ft.Row(
            visible=False,
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                self.edit_name,
                ft.IconButton(
                    icon=ft.icons.DONE_OUTLINE_OUTLINED,
                    icon_color=ft.colors.GREEN,
                    tooltip="Update To-Do",
                    on_click=self.save_clicked,
                ),
            ],
        )

    def build_detail_view(self):
        return ft.Container(
            content=ft.Column([
                
            #      ft.Container(  # Special row for volume bar
            #     content=self.volume_bar,
            #     height=10,  # Fixed height
            #     expand=True,  # Make it stretchable
            #     padding=ft.padding.only(left=10, right=10),
            #     alignment=ft.alignment.center,  # Center the volume bar
            #     # visible=False,  # Initially invisible
            # ),
                # ft.Row([
                ft.Row([
                    ft.IconButton(
                        icon=ft.icons.CALENDAR_TODAY,
                        tooltip="Set Due Date",
                        icon_color=ft.colors.BLACK,
                        on_click=self.show_date_picker,
                    ),
                    ft.IconButton(
                        icon=ft.icons.ACCESS_TIME,
                        tooltip="Set Alarm Time",
                        icon_color=ft.colors.BLACK,
                        on_click=self.set_alarm_time,
                    ),
                    self.record_button,
                    # self.volume_bar,  # Add volume bar here
                    self.play_pause_button,
                    self.description_button,
                    self.image_button,
                    self.search_descriptions,  # Added here
                ]),
                self.voice_notes_container,
                self.waveform,
                self.alarm_time_text,
                self.description_preview,
                self.descriptions_container,
            ]),
            bgcolor=ft.colors.GREY_900,
            opacity=0.5,
            padding=3,
            border_radius=ft.border_radius.all(5),
        )

    def create_more_options_menu(self):
        menu_items = [
            ft.PopupMenuItem(
                text="Unlock" if self.locked else "Lock",
                icon=ft.icons.LOCK_OPEN if self.locked else ft.icons.LOCK,
                on_click=self.toggle_lock
            ),
        ]
        
        if self.locked:
            menu_items.append(
                ft.PopupMenuItem(
                    text="Unlock with Password",
                    icon=ft.icons.PASSWORD,
                    on_click=self.show_unlock_dialog
                )
            )
        
        menu_items.extend([
            # ft.PopupMenuItem(text="Share", icon=ft.icons.SHARE, on_click=self.share_task),
            ft.PopupMenuItem(text="Duplicate", icon=ft.icons.CONTENT_COPY, on_click=self.duplicate_task),
            ft.PopupMenuItem(text="Delete", icon=ft.icons.DELETE, on_click=self.delete_clicked),
            # ft.PopupMenuItem(text="Share Task", icon=ft.icons.SHARE, on_click=self.share_task),
            # ft.PopupMenuItem(text="Set Due Date", icon=ft.icons.EVENT, on_click=self.set_due_date),
            # ft.PopupMenuItem(text="Make a QrCode", icon=ft.icons.QR_CODE, on_click=self.generate_qr_code),
            
            # ft.PopupMenuItem(text="Priority", icon=ft.icons.FLAG, on_click=self.show_priority_dialog),
            # ft.PopupMenuItem(text="Add Reminder", icon=ft.icons.ALARM, on_click=self.add_reminder),
            ft.PopupMenuItem(text="Add Attachment", icon=ft.icons.ATTACH_FILE, on_click=self.add_attachment),
            ft.PopupMenuItem(text="Add Tags", icon=ft.icons.LOCAL_OFFER, on_click=self.add_tags),
            # ft.PopupMenuItem(text="Export Task", icon=ft.icons.DOWNLOAD, on_click=self.export_task),
            # ft.PopupMenuItem(text="Add Subtask", icon=ft.icons.PLAYLIST_ADD, on_click=self.add_subtasks),
            ft.PopupMenuItem(text="Add Collaborator", icon=ft.icons.PERSON_ADD, on_click=self.add_collaborator),
            # ft.PopupMenuItem(text="Add Location", icon=ft.icons.LOCATION_ON, on_click=self.add_location),
            # ft.PopupMenuItem(text="Add Notes", icon=ft.icons.NOTE_ADD, on_click=self.add_notes),
            ft.PopupMenuItem(text="Add Time Estimate", icon=ft.icons.TIMER, on_click=self.add_time_estimate),
            # ft.PopupMenuItem(text="Add Project", icon=ft.icons.FOLDER, on_click=self.add_project),
            ft.PopupMenuItem(text="Set Deadline", icon=ft.icons.SCHEDULE, on_click=self.set_deadline),
            ft.PopupMenuItem(text="Set Progress", icon=ft.icons.TRENDING_UP, on_click=self.set_progress),
            ft.PopupMenuItem(text="Mind Map", icon=ft.icons.BUBBLE_CHART, on_click=self.create_mind_map),
            ft.PopupMenuItem(text="Voice Command", icon=ft.icons.RECORD_VOICE_OVER, on_click=self.add_voice_command),
            # ft.PopupMenuItem(text="Create Time            ft.PopupMenuItem(text="Make a            ft.PopupMenuItem(text="Biometric Lock", icon=ft.icons.FINGERPRINT, on_click=self.generate_biometric_lock),
            # ft.PopupMenuItem(text="Link Task", icon=ft.icons.LINK, on_click=self.link_task),
            ft.PopupMenuItem(text="Create Subtask", icon=ft.icons.SUBDIRECTORY_ARROW_RIGHT, on_click=self.create_subtask),        ])
        
        return ft.PopupMenuButton(
            icon=ft.icons.MORE_VERT,
            tooltip="More options",
            items=menu_items
        )
        
    def toggle_expand(self, e):
        self.expanded = not self.expanded
        self.expand_button.icon = ft.icons.EXPAND_LESS if self.expanded else ft.icons.EXPAND_MORE
        self.detail_view.visible = self.expanded
        self.update()

    def toggle_playback(self, voice_note):
        play_button, pause_button, resume_button = self.play_pause_buttons[voice_note]
        if not voice_note.is_playing:
            self.start_playback(voice_note)
            play_button.visible = False
            pause_button.visible = True
            resume_button.visible = False
        elif voice_note.is_paused:
            self.resume_playback(voice_note)
            play_button.visible = False
            pause_button.visible = True
            resume_button.visible = False
        else:
            self.pause_playback(voice_note)
            play_button.visible = False
            pause_button.visible = False
            resume_button.visible = True
        self.update()

    def start_playback(self, voice_note):
        voice_note.is_playing = True
        voice_note.is_paused = False
        voice_note.start_time = time.time()
        audio_to_play = voice_note.audio_data.flatten()[voice_note.playback_position:]
        sd.play(audio_to_play, voice_note.fs)
        self.update()

        threading.Thread(target=self.monitor_playback, args=(voice_note,), daemon=True).start()
        threading.Thread(target=self.update_playback_time, args=(voice_note,), daemon=True).start()

        self.play_pause_button.visible = False
        self.pause_button.visible = True
        self.resume_button.visible = False
        self.update()
        
    def pause_playback(self, voice_note):
        if sd.get_stream() is not None:
            elapsed_time = time.time() - voice_note.start_time
            voice_note.playback_position += int(elapsed_time * voice_note.fs)
        sd.stop()
        voice_note.is_playing = False
        voice_note.is_paused = True
        voice_note.pause_time = time.time()
        self.update_time_display(voice_note)
        self.update()

        self.play_pause_button.visible = False
        self.pause_button.visible = False
        self.resume_button.visible = True
        self.update()
        
    def resume_playback(self, voice_note):
        voice_note.is_playing = True
        voice_note.is_paused = False
        voice_note.start_time = time.time() - (voice_note.pause_time - voice_note.start_time)
        audio_to_play = voice_note.audio_data.flatten()[voice_note.playback_position:]
        sd.play(audio_to_play, voice_note.fs)
        self.update()

        threading.Thread(target=self.monitor_playback, args=(voice_note,), daemon=True).start()
        threading.Thread(target=self.update_playback_time, args=(voice_note,), daemon=True).start()

        play_button, pause_button, resume_button = self.play_pause_buttons[voice_note]
        play_button.visible = False
        pause_button.visible = True
        resume_button.visible = False
        self.update()
        
    def monitor_playback(self, voice_note):
        while voice_note.is_playing and sd.get_stream().active:
            time.sleep(0.1)

        if voice_note.is_playing:
            voice_note.is_playing = False
            voice_note.playback_position = 0
            voice_note.current_time = 0
            asyncio.run_coroutine_threadsafe(self.update_play_button(voice_note), self.page.loop)
            
    async def update_play_button(self, voice_note):
        play_button, pause_button, resume_button = self.play_pause_buttons[voice_note]
        play_button.visible = True
        pause_button.visible = False
        resume_button.visible = False
        self.update()

    def update_playback_time(self, voice_note):
        start_time = time.time()
        while voice_note.is_playing:
            voice_note.current_time = voice_note.playback_position / voice_note.fs + (time.time() - start_time)
            self.update_time_display(voice_note)
            time.sleep(0.1)  # Update every 100ms for smoother display
            
    def update_time_display(self, voice_note):
        for control in self.voice_notes_container.controls:
            if control.data == voice_note:
                time_display = control.content.controls[-1]  # Assuming the time display is the last control in the row
                current_time = self.format_time(voice_note.current_time)
                total_time = self.format_time(voice_note.duration)
                time_display.value = f"{current_time} / {total_time}"
                self.page.update()
                break

    def toggle_recording(self, e):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        try:
            self.record_generator = record_audio(fs=self.fs, device=self.input_device)
            self.record_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.record_thread.start()

            self.record_button.icon = ft.icons.STOP
            self.record_button.tooltip = "Stop Recording"
            self.volume_bar.visible = True  # Ensure volume bar is visible
            self.update()
        except Exception as e:
            print(f"Error during recording: {e}")
            self.is_recording = False
            self.volume_bar.visible = False  # Hide volume bar if recording fails
            self.update()

    def stop_recording(self):
        self.is_recording = False
        self.record_thread.join()

        self.record_button.icon = ft.icons.MIC
        self.record_button.tooltip = "Record Voice Note"
        self.volume_bar.visible = False  # Hide volume bar when recording stops

        if len(self.audio_data) > 0:
            audio_data = np.concatenate(self.audio_data)
            voice_note = VoiceNote(audio_data, self.fs)
            self.voice_notes.append(voice_note)
            self.add_voice_note_ui(voice_note)
        else:
            print("No audio data recorded")

        self.update()  # Update the UI to reflect changes

    # def _record_audio(self):
    #     while self.is_recording:
    #         chunk = next(self.record_generator)
    #         self.audio_data.append(chunk)
            
    #         # Update volume meter
    #         volume = np.abs(chunk).mean()
    #         self.volume_bar.value = volume
    #         self.volume_bar.update()
    
    def _record_audio(self):
        while self.is_recording:
            chunk = next(self.record_generator)
            self.audio_data.append(chunk)

            # Update volume meter
            volume = np.abs(chunk).mean()
            self.volume_bar.value = volume
            self.volume_bar.update()
            # Use asyncio.run_coroutine_threadsafe to update the UI from a separate thread
            # asyncio.run_coroutine_threadsafe(self.volume_bar.update(), self.page.loop)

    def add_voice_note_ui(self, voice_note):
        checkbox = ft.Checkbox(
            value=getattr(voice_note, 'is_important', False),
            on_change=lambda _: self.toggle_voice_note(voice_note),
            fill_color=ft.colors.BLUE,
            # shape=ft.CircleBorder(),
            scale=0.8,
        )
        star_icon = ft.IconButton(
            icon=ft.icons.STAR if getattr(voice_note, 'is_important', False) else ft.icons.STAR_BORDER,
            icon_color=ft.colors.AMBER if getattr(voice_note, 'is_important', False) else ft.colors.GREY_400,
            on_click=lambda _: self.toggle_voice_note(voice_note),
            icon_size=18,
        )
        play_button = ft.IconButton(
            icon=ft.icons.PLAY_ARROW,
            on_click=lambda _: self.toggle_playback(voice_note),
            icon_size=18,
            icon_color=ft.colors.BLUE,
        )
        pause_button = ft.IconButton(
            icon=ft.icons.PAUSE,
            on_click=lambda _: self.pause_playback(voice_note),
            icon_size=18,
            icon_color=ft.colors.RED,
        )
        resume_button = ft.IconButton(
            icon=ft.icons.PLAY_CIRCLE_FILLED,
            on_click=lambda _: self.resume_playback(voice_note),
            icon_size=18,
            icon_color=ft.colors.GREEN,
        )
        
        # Store the buttons for this voice note
        self.play_pause_buttons[voice_note] = (play_button, pause_button, resume_button)
        
        waveform = ft.Image(
            src_base64=generate_waveform(voice_note.audio_data.flatten(), voice_note.fs),
            fit=ft.ImageFit.FIT_WIDTH,
            height=30,
        )
        delete_button = ft.IconButton(
            icon=ft.icons.DELETE,
            on_click=lambda _: self.delete_voice_note(voice_note),
            icon_size=18,
        )
        
        total_time = self.format_time(voice_note.duration)
        time_display = ft.Text(f"00:00 / {total_time}", size=10, width=70)  # Fixed width

        left_buttons = ft.Row(
            [checkbox, star_icon, play_button, pause_button, resume_button],
            spacing=0,
            alignment=ft.MainAxisAlignment.START,
            tight=True,
        )

        voice_note_row = ft.Container(
            content=ft.Row([
                left_buttons,
                ft.Container(
                    content=waveform,
                    expand=True,  # This allows the waveform to expand and fill available space
                ),
                time_display,
                delete_button
            ], 
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            tight=True,
            ),
            bgcolor=ft.colors.AMBER_100 if getattr(voice_note, 'is_important', False) else ft.colors.BLUE_50,
            border=ft.border.all(1, ft.colors.AMBER) if getattr(voice_note, 'is_important', False) else ft.border.all(1, ft.colors.BLUE_200),
            border_radius=ft.border_radius.all(4),
            padding=5,
            margin=ft.margin.only(bottom=2),
        )
        voice_note_row.data = voice_note
        self.voice_notes_container.controls.append(voice_note_row)
        self.update()

    def toggle_playback(self, voice_note):
        play_button, pause_button, resume_button = self.play_pause_buttons[voice_note]
        if not voice_note.is_playing:
            self.start_playback(voice_note)
            play_button.icon = ft.icons.PAUSE
            play_button.icon_color = ft.colors.RED
            resume_button.icon_color = ft.colors.GREY_400
        elif voice_note.is_paused:
            self.resume_playback(voice_note)
            play_button.icon = ft.icons.PAUSE
            play_button.icon_color = ft.colors.RED
            resume_button.icon_color = ft.colors.GREY_400
        else:
            self.pause_playback(voice_note)
            play_button.icon = ft.icons.PLAY_ARROW
            play_button.icon_color = ft.colors.BLUE
            resume_button.icon_color = ft.colors.GREEN
        self.update()

    def start_playback(self, voice_note):
        voice_note.is_playing = True
        voice_note.is_paused = False
        voice_note.start_time = time.time()
        audio_to_play = voice_note.audio_data.flatten()[voice_note.playback_position:]
        sd.play(audio_to_play, voice_note.fs)
        self.update()

        threading.Thread(target=self.monitor_playback, args=(voice_note,), daemon=True).start()
        threading.Thread(target=self.update_playback_time, args=(voice_note,), daemon=True).start()

        play_button, pause_button, resume_button = self.play_pause_buttons[voice_note]
        play_button.icon = ft.icons.PAUSE
        play_button.icon_color = ft.colors.RED
        resume_button.icon_color = ft.colors.GREY_400
        self.update()

    def pause_playback(self, voice_note):
        if sd.get_stream() is not None:
            elapsed_time = time.time() - voice_note.start_time
            voice_note.playback_position += int(elapsed_time * voice_note.fs)
        sd.stop()
        voice_note.is_playing = False
        voice_note.is_paused = True
        voice_note.pause_time = time.time()
        self.update_time_display(voice_note)
        self.update()

        play_button, pause_button, resume_button = self.play_pause_buttons[voice_note]
        play_button.icon = ft.icons.PLAY_ARROW
        play_button.icon_color = ft.colors.BLUE
        resume_button.icon_color = ft.colors.GREEN
        self.update()

    def resume_playback(self, voice_note):
        voice_note.is_playing = True
        voice_note.is_paused = False
        voice_note.start_time = time.time() - (voice_note.pause_time - voice_note.start_time)
        audio_to_play = voice_note.audio_data.flatten()[voice_note.playback_position:]
        sd.play(audio_to_play, voice_note.fs)
        self.update()

        threading.Thread(target=self.monitor_playback, args=(voice_note,), daemon=True).start()
        threading.Thread(target=self.update_playback_time, args=(voice_note,), daemon=True).start()

        play_button, pause_button, resume_button = self.play_pause_buttons[voice_note]
        play_button.icon = ft.icons.PAUSE
        play_button.icon_color = ft.colors.RED
        resume_button.icon_color = ft.colors.GREY_400
        self.update()

    def monitor_playback(self, voice_note):
        while voice_note.is_playing and sd.get_stream().active:
            time.sleep(0.1)

        if voice_note.is_playing:
            voice_note.is_playing = False
            voice_note.playback_position = 0
            voice_note.current_time = 0
            asyncio.run_coroutine_threadsafe(self.update_play_button(voice_note), self.page.loop)

    async def update_play_button(self, voice_note):
        play_button, pause_button, resume_button = self.play_pause_buttons[voice_note]
        play_button.icon = ft.icons.PLAY_ARROW
        play_button.icon_color = ft.colors.BLUE
        resume_button.icon_color = ft.colors.GREY_400
        self.update()

    def update_playback_time(self, voice_note):
        start_time = time.time()
        while voice_note.is_playing:
            voice_note.current_time = voice_note.playback_position / voice_note.fs + (time.time() - start_time)
            self.update_time_display(voice_note)
            time.sleep(0.1)  # Update every 100ms for smoother display

    def update_time_display(self, voice_note):
        for control in self.voice_notes_container.controls:
            if control.data == voice_note:
                time_display = control.content.controls[-1]  # Assuming the time display is the last control in the row
                current_time = self.format_time(voice_note.current_time)
                total_time = self.format_time(voice_note.duration)
                time_display.value = f"{current_time} / {total_time}"
                self.page.update()
                break

    def format_time(self, seconds):
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"

    def toggle_voice_note(self, voice_note):
        voice_note.is_important = not getattr(voice_note, 'is_important', False)
        
        for control in self.voice_notes_container.controls:
            if control.data == voice_note:
                checkbox = control.content.controls[0]  # Assuming checkbox is the first control
                checkbox.value = voice_note.is_important
                checkbox.shape = ft.CircleBorder()  # Ensure the checkbox remains rounded
                
                if voice_note.is_important:
                    control.bgcolor = ft.colors.AMBER_100
                    control.border = ft.border.all(2, ft.colors.AMBER)
                else:
                    control.bgcolor = ft.colors.BLUE_50
                    control.border = ft.border.all(1, ft.colors.BLUE_200)
                
                # Update the star icon
                star_icon = control.content.controls[1]  # Assuming star icon is the second control
                star_icon.icon = ft.icons.STAR if voice_note.is_important else ft.icons.STAR_BORDER
                star_icon.icon_color = ft.colors.AMBER if voice_note.is_important else ft.colors.GREY_400
                
                break
        
        self.update()
        
        # Optional: Implement sorting based on importance
        self.sort_voice_notes()

    def delete_voice_note(self, voice_note):
        # Remove the voice note from the list if it exists
        if voice_note in self.voice_notes:
            self.voice_notes.remove(voice_note)
        
        # Remove the corresponding UI control
        for control in self.voice_notes_container.controls[:]:
            if control.data == voice_note:
                self.voice_notes_container.controls.remove(control)
                break
        
        self.update()

    def show_date_picker(self, e):
        if not self.due_date_picker:
            self.due_date_picker = ft.DatePicker(
                on_change=self.set_due_date,
                first_date=date.today(),
                last_date=date(2050, 12, 31)
            )
            self.page.overlay.append(self.due_date_picker)
            self.page.update()
        self.due_date_picker.open = True
        self.page.update()

    def set_due_date(self, e):
        if e.control.value:
            self.due_date = e.control.value.date()
            self.display_task.label = f"{self.task_name} (Due: {self.due_date.strftime('%Y-%m-%d')})"
        else:
            self.due_date = None
            self.display_task.label = self.task_name
        self.due_date_picker.open = False
        self.update()

    def edit_clicked(self, e):
        self.edit_name.value = self.task_name
        self.display_view.visible = False
        self.edit_view.visible = True
        self.update()

    def save_clicked(self, e):
        new_name = self.edit_name.value
        if new_name and new_name != self.task_name:
            self.task_name = new_name
            self.display_task.label = self.task_name
        self.display_view.visible = True
        self.edit_view.visible = False
        self.update()

    def status_changed(self, e):
        self.task_status_change(self)

    def delete_clicked(self, e):
        self.task_delete(self)

    def set_alarm_time(self, e):
        def open_picker():
            time_picker.pick_time()

        time_picker = ft.TimePicker(
            ref=self.alarm_time,
            on_change=self.set_alarm,
            on_dismiss=self.close_time_picker
        )
        self.page.overlay.append(time_picker)
        self.page.update()

        self.page.add(time_picker)

        threading.Timer(0.1, open_picker).start()

    def set_alarm(self, e):
        if self.alarm_time.current.value:
            selected_time = self.alarm_time.current.value.strftime("%H:%M")
            self.alarm_time_text.value = f"Alarm set for {selected_time}"
            self.alarm_time_text.style = ft.TextStyle(
                weight=ft.FontWeight.BOLD,
                color=ft.colors.GREEN
            )
            self.alarm_time_text.visible = True
            self.alarm_active = True
        else:
            self.alarm_time_text.value = "Alarm not set"
            self.alarm_time_text.style = None
            self.alarm_time_text.visible = False
            self.alarm_active = False
        self.page.update()

    def close_time_picker(self, e):
        self.page.overlay.clear()
        self.page.update()

    def check_alarm(self):
        while self.alarm_active:
            current_time = datetime.now().strftime("%H:%M")
            if self.alarm_time.current.value and self.alarm_time.current.value.strftime("%H:%M") == current_time:
                self.trigger_alarm()
                break
            time.sleep(1)

    def trigger_alarm(self):
        self.page.snack_bar = ft.SnackBar(ft.Text(f"Alarm triggered for task: {self.task_name}!"))
        self.page.snack_bar.open = True
        self.alarm_active = False
        self.alarm_time_text.value = "Alarm not set"
        self.alarm_time_text.visible = False
        self.page.update()

    
    def show_color_picker(self, e):
        color_picker = ColorPicker(
            width=300,
            color="#000000",  # Default color
        )

        def on_color_changed(color):
            self.current_color = color
            self.apply_formatting("color", color)

        color_picker.on_change = on_color_change        
        color_picker_dialog = ft.AlertDialog(
            title=ft.Text("Pick a color"),
            content=ft.Container(
                content=color_picker,
                padding=10,
            ),
            actions=[
                ft.TextButton("Done", on_click=lambda _: self.close_color_picker(color_picker_dialog)),
            ],
        )

        self.page.dialog = color_picker_dialog
        color_picker_dialog.open = True
        self.page.update()
    
    def on_color_changed(self, color):
        self.apply_formatting("color", color)

    def close_color_picker(self, dialog):
        dialog.open = False
        self.page.update()

    def apply_color_picker(self, color_picker, dialog):
        self.current_color = color_picker.color
        self.apply_formatting("color", self.current_color)
        self.close_color_picker(dialog)

    def add_description_clicked(self, e):
        self.show_description_dialog()

    def show_description_dialog(self, existing_description=None, edit_index=None):
        self.description_field = ft.TextField(
            multiline=True,
            min_lines=3,
            max_lines=10,
            value=existing_description if existing_description is not None else "",
            expand=True
        )

        save_button = ft.TextButton("Save", on_click=lambda _: self.save_description_and_close(edit_index))
        cancel_button = ft.TextButton("Cancel", on_click=self.close_description_dialog)

        description_dialog = ft.AlertDialog(
            title=ft.Text("Edit Description" if edit_index is not None else "Add Description"),
            content=ft.Column([self.description_field]),
            actions=[cancel_button, save_button],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.dialog = description_dialog
        description_dialog.open = True
        self.page.update()

    def save_description_and_close(self, edit_index=None):
        description_text = self.description_field.value
        if description_text:
            if edit_index is not None:  # Editing existing description
                if 0 <= edit_index < len(self.descriptions):
                    self.descriptions[edit_index] = description_text
                else:
                    print(f"Invalid index: {edit_index}")
            else:  # Adding new description
                self.descriptions.append(description_text)
            self.update_descriptions_ui()
        self.close_description_dialog()
        self.page.update()  # Update the entire page to reflect changes

    def close_description_dialog(self, e=None):
        if self.page.dialog:
            self.page.dialog.open = False
            self.page.dialog = None
        self.page.update()
        
    def update_descriptions_ui(self):
        self.descriptions_container.controls.clear()
        for index, desc in enumerate(self.descriptions):
            desc_text = ft.Text(desc[:50] + "..." if len(desc) > 50 else desc, expand=True)
            edit_button = ft.IconButton(
                icon=ft.icons.MODE_EDIT,
                icon_size=16,
                tooltip="Edit Description",
                on_click=lambda _, i=index: self.edit_description(i)
            )
            delete_button = ft.IconButton(
                icon=ft.icons.DELETE,
                icon_size=16,
                tooltip="Delete Description",
                icon_color=ft.colors.RED,
                on_click=lambda _, i=index: self.delete_description(i)
            )
            description_row = ft.Container(
                content=ft.Row([edit_button,desc_text, delete_button]),
                bgcolor=ft.colors.BLUE_50,
                border=ft.border.all(1, ft.colors.BLUE_200),
                border_radius=ft.border_radius.all(8),
                # height=20,
                margin=ft.margin.only(bottom=5),
                # padding=ft.padding.only(left=10, right=10, top=5, bottom=5),
            )
            self.descriptions_container.controls.append(description_row)
        self.update()  # This updates the task view
        
    def delete_description(self, index):
        if 0 <= index < len(self.descriptions):
            self.descriptions.pop(index)
            self.update_descriptions_ui()
        else:
            print(f"Invalid index: {index}")

    def update_formatting(self, e):
        self.description = self.description_area.value
        self.description_area.update()

    def apply_formatting(self, tag, value=None):
        self.formatting.append((tag, value))
        self.update_description_preview()

    def update_description(self, e):
        self.description = self.description_field.value
        self.update_description_preview()
    def update_description_preview(self):
        if self.description:
            formatted_preview = self.render_formatted_text(self.description)
            self.description_preview.content = formatted_preview
            self.description_preview.visible = True
            self.description_button.icon = ft.icons.TEXT_SNIPPET
            self.description_button.icon_color = ft.colors.BLUE
            self.description_button.tooltip = "Edit Description"
        else:
            self.description_preview.visible = False
            self.description_button.icon = ft.icons.DESCRIPTION_OUTLINED
            

            self.description_button.icon_color = None
            self.description_button.tooltip = "Add Description"
        self.update()

    def render_formatted_text(self, text):
        formatted_text = ft.Column(spacing=0)
        current_style = ft.TextStyle()
        current_align = ft.TextAlign.LEFT

        for tag, value in self.formatting:
            if tag == 'b':
                current_style.weight = ft.FontWeight.BOLD
            elif tag == 'u':
                current_style.decoration = ft.TextDecoration.UNDERLINE
            elif tag == 'color':
                current_style.color = value
            elif tag == 'size':
                if value == 'small':
                    current_style.size = 12
                elif value == 'medium':
                    current_style.size = 16
                elif value == 'large':
                    current_style.size = 20
            elif tag == 'align':
                if value == 'left':
                    current_align = ft.TextAlign.LEFT
                elif value == 'center':
                    current_align = ft.TextAlign.CENTER
                elif value == 'right':
                    current_align = ft.TextAlign.RIGHT

        formatted_text.controls.append(ft.Text(
            text,
            style=current_style,
            text_align=current_align
        ))

        return formatted_text

    def toggle_bold(self, e):
        self.apply_formatting("b")

    def set_alignment(self, align):
        self.apply_formatting("align", align)

    def set_text_size(self, e):
        size = e.control.value
        self.apply_formatting("size", size)

    def toggle_underline(self, e):
        self.apply_formatting("u")
    
    def show_image_dialog(self, e):
        def add_image(e):
            file_picker.pick_files(allow_multiple=False)

        def file_picker_result(e: ft.FilePickerResultEvent):
            if e.files:
                image_path = e.files[0].path
                self.set_task_image(image_path)

        def set_background_color(e):
            color_picker = ColorPicker(
                width=300,
                color="#000000",  # Default color
                on_change=lambda color: self.set_task_background(color)
            )
            color_dialog = ft.AlertDialog(
                title=ft.Text("Pick a background color"),
                content=color_picker,
                actions=[
                    ft.TextButton("Done", on_click=lambda _: self.close_dialog(color_dialog)),
                ],
            )
            self.page.dialog = color_dialog
            color_dialog.open = True
            self.page.update()

        file_picker = ft.FilePicker(on_result=file_picker_result)
        self.page.overlay.append(file_picker)

        dialog = ft.AlertDialog(
            title=ft.Text("Add Image or Background Color"),
            content=ft.Column([
                ft.ElevatedButton("Add Image", on_click=add_image),
                ft.ElevatedButton("Set Background Color", on_click=set_background_color),
            ]),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: self.close_dialog(dialog)),
            ],
        )

        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

        
    def set_task_image(self, image_path):
        img = Image.open(image_path)
        img = img.convert('RGBA')  # Ensure image has an alpha channel
        
        # Create a new image with a white background
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        background.paste(img, (0, 0), img)
        
        buffered = io.BytesIO()
        background.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        self.task_background = ft.Image(src_base64=img_str, fit=ft.ImageFit.COVER, width=600, height=200)
        self.update_background()
        self.close_dialog(self.page.dialog)
        
    # def set_task_background(self, color):
    #     self.drop_container.bgcolor = color
    #     self.close_dialog(self.page.dialog)
    #     self.update()
    def set_task_background(self, color):
        self.task_background = color
        self.update_background()
        self.close_dialog(self.page.dialog)
                
    def close_dialog(self, dialog):
        dialog.open = False
        self.page.update()
        
    def set_priority(self, e):
        self.current_priority = e.control.text
        self.update_task_color()

    def update_task_color(self):
        color = self.priority_colors[self.current_priority]
        self.display_task.label_style = ft.TextStyle(color=color, weight=ft.FontWeight.BOLD)
        self.priority_dropdown.icon_color = color
        self.update()
    
    def update_background(self):
        if isinstance(self.task_background, ft.Image):
            self.drop_container.image_src = self.task_background.src_base64
            self.drop_container.image_fit = ft.ImageFit.COVER
        else:
            self.drop_container.bgcolor = self.task_background
            self.drop_container.image_src = None

        # Ensure content is visible over the background
        for control in self.drop_container.content.controls:
            if isinstance(control, ft.Container):
                control.bgcolor = ft.colors.with_opacity(0.7, ft.colors.WHITE)

        self.update()


# End of VoiceTask class


class TodoApp(ft.UserControl):
    def __init__(self):
        super().__init__()
        self.new_task = ft.TextField(
            hint_text="What needs to be done?",
            expand=True,
            bgcolor=ft.colors.GREY_900,
            border_radius=8,
            border=ft.border.all(1, ft.colors.LIME_700),
            color=ft.colors.WHITE,
            hint_style=ft.TextStyle(color=ft.colors.GREY_400),
            prefix_icon=ft.icons.ADD_TASK,
            suffix_icon=ft.icons.CHECK_CIRCLE_OUTLINE,
            on_submit=self.add_clicked,
            focused_border_color=ft.colors.BLUE_400,
            focused_bgcolor=ft.colors.GREY_800,
            cursor_color=ft.colors.BLUE_200,
            content_padding=10,
            # on_submit=self.add_task
        )
        self.search_field = ft.TextField(
            hint_text="Find tasks...",
            on_change=self.search_tasks,
            prefix_icon=ft.icons.SEARCH,
            expand=True,
            visible=False  # Initially hide the search field
        )
        self.search_field.on_suffix_click = self.clear_search
        self.search_button = ft.IconButton(
            icon=ft.icons.SEARCH,
            tooltip="Search",
            on_click=self.toggle_search,
            # suffix=ft.IconButton(icon=ft.icons.CLOSE, on_click=self.clear_search),
            # on_submit=self.search_tasks,
        )
        
        self.tasks = ft.Column()
        self.filter = ft.Tabs(
            selected_index=0,
            on_change=self.tabs_changed,
            tabs=[ft.Tab(text="all"), ft.Tab(text="active"), ft.Tab(text="completed")],
        )
        self.items_left = ft.Text("0 items left")
        self.theme_switch = ft.Switch(label="Sombre", on_change=self.theme_changed)
        self.input_device = None
        self.output_device = None
        self.fs = 44100
        
        self.dashboard_dialog = None
        self.create_dashboard_dialog()
    #------------------------------------------------------

    def build(self):
        return ft.Column(
            width=600,
            controls=[
                ft.Row(
                    [
                        ft.Text(
                            value="LeManager",
                            style=ft.TextThemeStyle.HEADLINE_MEDIUM,
                            color=ft.colors.BLUE,
                            weight=ft.FontWeight.BOLD
                        ),
                        self.theme_switch,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Row(
                    controls=[
                        self.new_task,
                        self.search_field,
                        self.search_button,
                        ft.FloatingActionButton(
                            icon=ft.icons.ADD,
                            on_click=self.add_clicked,
                            tooltip="Add new task"  # This is the new line
                        ),
                    ],
                ),
                
                self.filter,
                self.tasks,
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        self.items_left,
                        ft.OutlinedButton(
                            text="Clear completed", on_click=self.clear_completed_clicked
                        ),
                    ],
                ),
            ],
        )
    
    def toggle_search(self, e):
        self.search_field.visible = not self.search_field.visible
        self.search_button.icon = ft.icons.CLOSE if self.search_field.visible else ft.icons.SEARCH
        self.search_button.tooltip = "Close search" if self.search_field.visible else "Search tasks"
        if not self.search_field.visible:
            self.search_field.value = ""  # Clear search when hiding
            self.search_tasks(None)  # Reset search results
        self.update()
    
    def clear_search(self, e):
        self.search_field.value = ""
        self.search_field.suffix_icon = None
        self.search_tasks(None)  # Reset search results
        self.update()
    def search_tasks(self, e):
        search_term = self.search_field.value.lower()
        for task in self.tasks.controls:
            task.visible = self.task_matches_search(task, search_term)
        self.update()    
    def add_clicked(self, e):
        print("Add button clicked")  # Debug print
        if self.new_task.value:
            print(f"Adding new task: {self.new_task.value}")  # Debug print
            task = VoiceTask(self.page, self.new_task.value, self.task_delete, self.task_status_change, self.tasks, self.handle_dismissal)
            task.input_device = self.input_device
            task.fs = self.fs
            self.tasks.controls.append(task)
            self.new_task.value = ""
            self.update()
            print("Task added and UI updated")  # Debug print
        else:
            print("No task text entered")  # Debug print

    def handle_dismissal(self, e):
        print("Dismissal handled")
    
    def create_dashboard_dialog(self):
        pie_chart = self.create_pie_chart()
        bar_chart = self.create_bar_chart()
        line_chart = self.create_line_chart()

        self.dashboard_dialog = ft.AlertDialog(
            title=ft.Text("Dashboard"),
            content=ft.Tabs(
                selected_index=0,
                animation_duration=300,
                tabs=[
                    ft.Tab(text="Pie Chart", content=pie_chart),
                    ft.Tab(text="Bar Chart", content=bar_chart),
                    ft.Tab(text="Line Chart", content=line_chart),
                ],
                expand=1
            ),
            actions=[
                ft.TextButton("Close", on_click=self.close_dashboard_dialog),
            ],
        )

    def create_pie_chart(self):
        fig, ax = plt.subplots()
        tasks = [task for task in self.tasks.controls if isinstance(task, VoiceTask)]
        statuses = ['Completed', 'Active']
        sizes = [
            len([task for task in tasks if task.display_task.value]),
            len([task for task in tasks if not task.display_task.value])
        ]
        
        # Check if there are any tasks
        if sum(sizes) == 0:
            ax.text(0.5, 0.5, 'No tasks', horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
        else:
            # Remove any zero values to avoid the error
            non_zero_sizes = [size for size in sizes if size > 0]
            non_zero_statuses = [status for status, size in zip(statuses, sizes) if size > 0]
            
            ax.pie(non_zero_sizes, labels=non_zero_statuses, autopct='%1.1f%%', startangle=90)
        
        ax.axis('equal')
        
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)

        return ft.Image(src_base64=img_base64, width=400, height=300)

    def create_bar_chart(self):
        # Create and return a bar chart
        fig, ax = plt.subplots()
        tasks = [task for task in self.tasks.controls if isinstance(task, VoiceTask)]
        priorities = list(tasks[0].priority_colors.keys()) if tasks else []
        counts = [len([task for task in tasks if task.current_priority == priority]) for priority in priorities]
        
        ax.bar(priorities, counts)
        ax.set_ylabel('Number of Tasks')
        ax.set_title('Tasks by Priority')
        
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)

        return ft.Image(src_base64=img_base64, width=400, height=300)

    def create_line_chart(self):
        # Create and return a line chart
        fig, ax = plt.subplots()
        tasks = [task for task in self.tasks.controls if isinstance(task, VoiceTask)]
        tasks_with_due_dates = [task for task in tasks if task.due_date]
        tasks_with_due_dates.sort(key=lambda x: x.due_date)
        
        dates = [task.due_date for task in tasks_with_due_dates]
        cumulative_tasks = list(range(1, len(dates) + 1))
        
        ax.plot(dates, cumulative_tasks)
        ax.set_xlabel('Due Date')
        ax.set_ylabel('Cumulative Number of Tasks')
        ax.set_title('Task Accumulation Over Time')
        
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)

        return ft.Image(src_base64=img_base64, width=400, height=300)

    def show_dashboard_dialog(self, e):
        self.create_dashboard_dialog()  # Recreate the dialog to update the charts
        self.page.dialog = self.dashboard_dialog
        self.dashboard_dialog.open = True
        self.page.update()

    def close_dashboard_dialog(self, e):
        self.dashboard_dialog.open = False
        self.page.update()
    #------------------------------------------------------
    def task_delete(self, task):
        print(f"Deleting task: {task.task_name}")  # Debug print
        self.tasks.controls.remove(task)
        self.update()

    def task_status_change(self, task):
        self.update()

    def tabs_changed(self, e):
        self.update()

    def clear_completed_clicked(self, e):
        for task in self.tasks.controls[:]:
            if task.display_task.value:
                self.task_delete(task)

    def update(self):
        status = self.filter.tabs[self.filter.selected_index].text
        search_term = self.search_field.value.lower()
        count = 0
        for task in self.tasks.controls:
            task.visible = (
                (status == "all"
                or (status == "active" and not task.display_task.value)
                or (status == "completed" and task.display_task.value))
                and task.matches_search(search_term)
            )
            if not task.display_task.value:
                count += 1

            if task.due_date and task.due_date < date.today() and not task.display_task.value:
                task.display_task.style = ft.TextStyle(color=ft.colors.RED)
            else:
                task.display_task.style = None

        self.items_left.value = f"{count} active item(s) left"
        
        if self.dashboard_dialog and self.dashboard_dialog.open:
            self.create_dashboard_dialog()  # Update the charts
            self.page.dialog = self.dashboard_dialog
            
        super().update()

    def theme_changed(self, e):
        self.page.theme_mode = (
            ft.ThemeMode.DARK
            if self.theme_switch.value
            else ft.ThemeMode.LIGHT
        )
        self.page.update()

    def task_matches_search(self, task, search_term):
        if search_term in task.task_name.lower():
            return True
        for voice_note in task.voice_notes:
            if search_term in voice_note.audio_data.tobytes().decode('utf-8', errors='ignore').lower():
                return True
        for description in task.descriptions:
            if search_term in description.lower():
                return True
        return False

def main(page: ft.Page):
    # page.debug = True
    page.title = "LeManager M App"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.theme_mode = ft.ThemeMode.LIGHT
    todo = TodoApp()
    
    def handle_dismissal(e):
        print("Drawer dismissed")

    def handle_change(e):
        if e.control.selected_index == 0:  # Dashboard
            todo.show_dashboard_dialog(e)
        print(f"Selected destination: {e.control.selected_index}")

    def open_drawer(e):
        page.drawer.open = True
        page.update()

    app_logo = ft.Image(
        src="assets/icon.png",  # Local image file in assets folder
        width=100,
        height=100,
        fit=ft.ImageFit.CONTAIN
    )

    drawer = ft.NavigationDrawer(
        on_dismiss=handle_dismissal,
        on_change=handle_change,
        controls=[
            ft.Container(
                
                content=ft.Column([
                    # app_logo,
                    ft.Icon(
                        name=ft.icons.TRENDING_UP,
                        size=130,
                        color=ft.colors.GREEN,
                    ),
                    ft.Text("LeManager", size=24, weight=ft.FontWeight.BOLD),
                    ft.Text("Organize tes taches sans effort !...", size=14, color=ft.colors.GREY_400),
                ]),
                padding=20,
                alignment=ft.alignment.center,
            ),
            ft.Divider(thickness=2),
            ft.NavigationDrawerDestination(
                icon_content=ft.Icon(icons.HOME_OUTLINED),
                label="Dashboard",
                selected_icon=icons.HOME,
            ),
            ft.NavigationDrawerDestination(
                icon_content=ft.Icon(icons.LIST_ALT_OUTLINED),
                label="All Tasks",
                selected_icon=icons.LIST_ALT,
            ),
            ft.NavigationDrawerDestination(
                icon_content=ft.Icon(icons.MIC_OUTLINED),
                label="Voice Notes",
                selected_icon=icons.MIC,
            ),
            
            ft.NavigationDrawerDestination(
                icon_content=ft.Icon(icons.CALENDAR_TODAY_OUTLINED),
                label="Calendar",
                selected_icon=icons.CALENDAR_TODAY,
            ),
            ft.Divider(thickness=1),
            ft.NavigationDrawerDestination(
                icon_content=ft.Icon(icons.SETTINGS_OUTLINED),
                label="Settings",
                selected_icon=icons.SETTINGS,
            ),
            ft.NavigationDrawerDestination(
                icon_content=ft.Icon(icons.HELP_OUTLINE),
                label="Help & Support",
                selected_icon=icons.HELP,
            ),
            ft.Container(height=20),  # Spacer
            ft.Row([
                ft.IconButton(icon=icons.BRIGHTNESS_6, tooltip="Toggle Theme"),
                ft.IconButton(icon=icons.NOTIFICATIONS, tooltip="Notifications"),
                ft.IconButton(icon=icons.ACCOUNT_CIRCLE, tooltip="Account"),
            ], alignment=ft.MainAxisAlignment.CENTER),
            ft.Container(
                content=ft.Text("v1.0.0", size=12, color=ft.colors.GREY_400),
                alignment=ft.alignment.center,
                margin=ft.margin.only(top=10)
            ),
        ],
    )

    page.drawer = drawer

    page.bottom_appbar = ft.BottomAppBar(
        bgcolor=ft.colors.BLUE,
        shape=ft.NotchShape.CIRCULAR,
        content=ft.Row(
            controls=[
                ft.IconButton(icon=icons.MENU, icon_color=ft.colors.WHITE, on_click=open_drawer),
                ft.Container(expand=True),
                ft.IconButton(icon=icons.ADD_TASK, icon_color=ft.colors.WHITE, tooltip="Quick Add Task"),
                ft.IconButton(icon=icons.SEARCH, icon_color=ft.colors.WHITE, tooltip="Search Tasks"),
                ft.IconButton(icon=icons.MORE_VERT, icon_color=ft.colors.WHITE, tooltip="More Options"),
            ]
        ),
    )

    todo = TodoApp()

    devices = get_audio_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    output_devices = [d for d in devices if d['max_output_channels'] > 0]
    
    main_input_device = next((d for d in input_devices if "default" in d['name'].lower() or "main" in d['name'].lower()), input_devices[0] if input_devices else None)
    main_output_device = next((d for d in output_devices if "default" in d['name'].lower() or "main" in d['name'].lower()), output_devices[0] if output_devices else None)
    
    if main_input_device:
        print(f"Main input device detected: {main_input_device['name']}")
    if main_output_device:
        print(f"Main output device detected: {main_output_device['name']}")

    input_dropdown = ft.Dropdown(
        label="Input Device",
        options=[ft.dropdown.Option(f"{d['name']} (Index: {d['index']})") for d in input_devices],
        width=300,
        icon=ft.icons.MIC,
    )

    output_dropdown = ft.Dropdown(
        label="Output Device",
        options=[ft.dropdown.Option(f"{d['name']} (Index: {d['index']})") for d in output_devices],
        width=300,
        icon=ft.icons.SPEAKER,
    )

    sample_rate_dropdown = ft.Dropdown(
        label="Sample Rate",
        options=[
            ft.dropdown.Option("11025 Hz"),
            ft.dropdown.Option("22050 Hz"),
            ft.dropdown.Option("44100 Hz"),
            ft.dropdown.Option("48000 Hz"),
            ft.dropdown.Option("96000 Hz"),
            ft.dropdown.Option("192000 Hz"),
        ],
        width=300,
        icon=ft.icons.SPEED,
    )

    def on_input_change(e):
        selected_index = int(input_dropdown.value.split("Index: ")[-1][:-1])
        todo.input_device = selected_index
        for task in todo.tasks.controls:
            task.input_device = selected_index

    def on_output_change(e):
        selected_index = int(output_dropdown.value.split("Index: ")[-1][:-1])
        todo.output_device = selected_index

    def on_sample_rate_change(e):
        selected_rate = int(sample_rate_dropdown.value.split()[0])
        todo.fs = selected_rate
        for task in todo.tasks.controls:
            task.fs = selected_rate

    input_dropdown.on_change = on_input_change
    output_dropdown.on_change = on_output_change
    sample_rate_dropdown.on_change = on_sample_rate_change

    page.add(
        ft.Column([
            input_dropdown,
            output_dropdown,
            sample_rate_dropdown,
            todo,  # This is the only place where tasks should be managed
        ]),
        page.bottom_appbar,
    )

    page.update()

if __name__ == "__main__":
    # ft.app(target=main)
    ft.app(target=main, assets_dir="assets", )
    # ft.app(target=main, port=8080, view=ft.WEB_BROWSER, assets_dir="assets")
    # ft.app(
    #     target=main,
    #     port=8080,
    #     view=ft.AppView.WEB_BROWSER,
    #     host="0.0.0.0",
    #     web_renderer="html"
    # )
