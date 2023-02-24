import customtkinter
import CalculateData

# sets the appearance mode to what the system appearance mode is (light or dark)
customtkinter.set_appearance_mode("System")

# set the default color theme to dark-blue
customtkinter.set_default_color_theme("blue")


# creates a class function for the creation of the GUI
class App(customtkinter.CTk):
    # create initialize function for the GUI
    def __init__(self):
        # allows it to call itself
        super().__init__()

        # sets the title for the gui
        self.title("Shawn Ruby C964")
        # sets the base resolution for the GUI
        self.geometry(f"{1100}x{700}")

        # configure grid layout (2x2)
        self.grid_columnconfigure(0, weight=70)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=3)

        # create frame that contains the search box and formats the grid placement
        self.movie_search_frame = customtkinter.CTkFrame(master=self, corner_radius=15)
        self.movie_search_frame.grid(row=1, column=0, padx=(5, 5), pady=(0, 5), sticky="nswe")

        # creates a label and sets the font, and size and formats the grid placement
        self.movie_title_label = customtkinter.CTkLabel(master=self.movie_search_frame, text="Type a movie:",
                                                        font=customtkinter.CTkFont("Times new Roman", size=24))
        self.movie_title_label.grid(row=1, column=1, padx=(10, 10), pady=(10, 5), sticky="nsew")

        # places the label at a specific placement and centers it
        self.movie_title_label.place(relx=0.43, rely=0.15, anchor="c")

        # creates a text box and formats the grid placement
        self.movie_entry = customtkinter.CTkEntry(master=self.movie_search_frame, placeholder_text="Movie Name")
        self.movie_entry.grid(row=1, column=1, padx=(10, 10), pady=(10, 5), sticky="nsew")

        # places the textbox at a specific placement and centers it
        self.movie_entry.place(relx=0.60, rely=0.15, anchor="c")

        # creates a label instructing the user to select and preferences they want to take into consideration
        self.preferences_label = customtkinter.CTkLabel(master=self.movie_search_frame, text='Select any preferences:',
                                                        font=customtkinter.CTkFont("Times new Roman", size=24))
        self.preferences_label.place(relx=0.5, rely=0.25, anchor="c")

        # creates the 4 checkboxes for the movie preferences and will call the create preference list function
        self.keywords_checkbox = customtkinter.CTkCheckBox(master=self.movie_search_frame,
                                                           command=self.create_preferences_list, text='Keywords')
        self.keywords_checkbox.place(relx=0.15, rely=0.40, anchor="c")
        self.cast_checkbox = customtkinter.CTkCheckBox(master=self.movie_search_frame,
                                                       command=self.create_preferences_list, text='Cast')
        self.cast_checkbox.place(relx=0.40, rely=0.40, anchor="c")
        self.director_checkbox = customtkinter.CTkCheckBox(master=self.movie_search_frame,
                                                           command=self.create_preferences_list, text='Director')
        self.director_checkbox.place(relx=0.65, rely=0.40, anchor="c")
        self.genres_checkbox = customtkinter.CTkCheckBox(master=self.movie_search_frame,
                                                         command=self.create_preferences_list, text='Genres')
        self.genres_checkbox.place(relx=0.90, rely=0.40, anchor="c")

        # create sidebar frame and configures the grid placement
        self.sidebar_frame = customtkinter.CTkFrame(master=self, corner_radius=15)
        self.sidebar_frame.grid(row=0, column=1, rowspan=2, padx=(0, 5), pady=(5, 5), sticky="nsew")

        # configures a placeholder at row 4
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # configures the label at the top of the sidebar and configures the grid placement
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Select how you \nwant to view the data",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # creates 3 buttons for each visual representation and configures the grid for them. If pressed,
        # it will run a function
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.search_movie_and_plot,
                                                        text="Plot Graph")
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=self.search_movie_and_bar,
                                                        text="Bar Chart")
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, command=self.search_movie_and_pie,
                                                        text="Pie Chart")
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)

        # creates a label for the appearance mode options and configures the grid placement
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:",
                                                            anchor="center")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))

        # creates a combo box that allows the user to switch between light mode and dark mode
        self.appearance_mode_option_menu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_option_menu.grid(row=6, column=0, padx=20, pady=(10, 10))

        # creates a label that allows the user to change the UI Scaling
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))

        # creates a combo box to change the ui scaling based on what the user picks
        self.scaling_option_menu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                               values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_option_menu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # creates the title of the GUI frame
        self.title_frame = customtkinter.CTkFrame(master=self, corner_radius=15)
        self.title_frame.grid(row=0, column=0, columnspan=1, rowspan=1, padx=(5, 5), pady=(5, 5), sticky="nsew")

        # creates 2 logos for the title frame
        self.logo_label = customtkinter.CTkLabel(self.title_frame, text="Movie Recommendation",
                                                 font=customtkinter.CTkFont("Times new Roman", size=40, weight="bold"))
        self.information_label = customtkinter.CTkLabel(self.title_frame, text="Search any movie to view "
                                                                               "recommendations",
                                                        font=customtkinter.CTkFont("Times New Roman",
                                                                                   size=28, weight="bold"))

        # places the 2 labels within the frame and centers them
        self.logo_label.place(relx=0.5, rely=0.4, anchor="c")
        self.information_label.place(relx=0.5, rely=0.65, anchor="c")

    # this function is what is called to change the appearance to dark or light mode
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    # this function is what is called to change the UI Scaling
    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    # this function will retrieve the movie name in text box and runs the function for the movie name
    def search_movie_and_plot(self):
        movie_name = self.movie_entry.get()
        CalculateData.get_data_and_plot(movie_name)

    # this function will retrieve the movie name in text box and runs the function for the movie name
    def search_movie_and_bar(self):
        movie_name = self.movie_entry.get()
        CalculateData.get_data_and_bar(movie_name)

    # this function will retrieve the movie name in text box and runs the function for the movie name
    def search_movie_and_pie(self):
        movie_name = self.movie_entry.get()
        CalculateData.get_data_and_pie(movie_name)

    # create the preferences list from what checkboxes are selected
    def create_preferences_list(self):
        # creates empty list
        preference_list = []
        # checks if any of the 4 buttons are toggled. This will add the dataframe column name to a list and will be
        # returned to the CalculateData file for processing
        if self.keywords_checkbox.get():
            preference_list.append('keywords')
        if self.cast_checkbox.get():
            preference_list.append('cast')
        if self.director_checkbox.get():
            preference_list.append('director')
        if self.genres_checkbox.get():
            preference_list.append('genres')
        CalculateData.get_preferences_list(preference_list)


# this will run the GUI
if __name__ == "__main__":
    app = App()
    # keeps GUI open
    app.mainloop()
