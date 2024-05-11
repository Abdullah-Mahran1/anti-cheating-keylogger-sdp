import tkinter as tk
from ftplib import FTP
from io import BytesIO

class FTPTextEditor:
    def _init_(self, master):
        self.master = master
        master.title("FTP Text Editor")

        self.ftp_server_label = tk.Label(master, text="FTP Server:")
        self.ftp_server_label.grid(row=0, column=0)
        self.ftp_server_entry = tk.Entry(master)
        self.ftp_server_entry.grid(row=0, column=1)

        self.username_label = tk.Label(master, text="Username:")
        self.username_label.grid(row=1, column=0)
        self.username_entry = tk.Entry(master)
        self.username_entry.grid(row=1, column=1)

        self.password_label = tk.Label(master, text="Password:")
        self.password_label.grid(row=2, column=0)
        self.password_entry = tk.Entry(master, show="*")
        self.password_entry.grid(row=2, column=1)

        self.list_button = tk.Button(master, text="List Files", command=self.list_files)
        self.list_button.grid(row=3, column=0, columnspan=2)

        self.files_listbox = tk.Listbox(master)
        self.files_listbox.grid(row=4, column=0, columnspan=2)

        self.retrieve_button = tk.Button(master, text="Retrieve Message", command=self.retrieve_message)
        self.retrieve_button.grid(row=5, column=0, columnspan=2)

        self.text_editor = tk.Text(master)
        self.text_editor.grid(row=6, column=0, columnspan=2)

        self.save_button = tk.Button(master, text="Save Content", command=self.save_content)
        self.save_button.grid(row=7, column=0, columnspan=2)

        # Initialize a variable to store the selected file
        self.selected_file = None

        # Start the update loop
        self.update_content_loop()

    def list_files(self):
        ftp_server = self.ftp_server_entry.get()
        username = self.username_entry.get()
        password = self.password_entry.get()

        try:
            with FTP(ftp_server) as ftp:
                ftp.login(username, password)
                files = ftp.nlst()
                self.files_listbox.delete(0, tk.END)
                for file in files:
                    self.files_listbox.insert(tk.END, file)
        except Exception as e:
            self.show_message("Error: " + str(e))

    def retrieve_message(self):
        self.selected_file = self.files_listbox.get(tk.ACTIVE)
        if self.selected_file.endswith('.txt'):
            # If the selected file is a text file, retrieve its content
            self.update_content()
        else:
            # If the selected file is a directory, list its contents
            ftp_server = self.ftp_server_entry.get()
            username = self.username_entry.get()
            password = self.password_entry.get()
            try:
                with FTP(ftp_server) as ftp:
                    ftp.login(username, password)
                    files = ftp.nlst(self.selected_file)
                    self.files_listbox.delete(0, tk.END)
                    for file in files:
                        self.files_listbox.insert(tk.END, file)
            except Exception as e:
                self.show_message("Error: " + str(e))

    def update_content(self):
        ftp_server = self.ftp_server_entry.get()
        username = self.username_entry.get()
        password = self.password_entry.get()

        try:
            with FTP(ftp_server) as ftp:
                ftp.login(username, password)
                bio = BytesIO()
                ftp.retrbinary('RETR ' + self.selected_file, bio.write)
                content = bio.getvalue().decode()
                self.text_editor.delete("1.0", tk.END)
                self.text_editor.insert(tk.END, content)
                self.show_message("Message retrieved successfully.")
        except Exception as e:
            self.show_message("Error: " + str(e))

    def save_content(self):
        ftp_server = self.ftp_server_entry.get()
        username = self.username_entry.get()
        password = self.password_entry.get()
        selected_file = self.files_listbox.get(tk.ACTIVE)
        content = self.text_editor.get("1.0", tk.END)

        try:
            with FTP(ftp_server) as ftp:
                ftp.login(username, password)
                bio = BytesIO(content.encode())
                ftp.storbinary('STOR ' + selected_file, bio)
                self.show_message("Content saved successfully.")
        except Exception as e:
            self.show_message("Error: " + str(e))

    def show_message(self, message):
        self.message_label = tk.Label(self.master, text=message)
        self.message_label.grid(row=8, column=0, columnspan=2)

    def update_content_loop(self):
        # Update content every 5 seconds
        self.master.after(5000, self.update_content_loop)
        # Check if a file is selected and update its content
        if self.selected_file:
            self.update_content()

if _name_ == "_main_":
    root = tk.Tk()
    app = FTPTextEditor(root)
    root.mainloop()