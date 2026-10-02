"""Test prompts for the three new rendering tools: SVG, Mermaid, and Vega-Lite."""

WEB_TEST_PROMPTS = {
    # -- SVG icons and graphics --
    "svg_house": "Draw a house icon as SVG: red triangle roof with polygon points at (20,110) (100,30) (180,110), tan rectangle body at x=20 y=110 width=160 height=90, brown door rect at x=75 y=155 width=40 height=45, blue window rect at x=30 y=125 width=35 height=30.",
    "svg_traffic_light": "Draw a traffic light icon with three colored circles: red on top, yellow in middle, green at bottom, inside a dark rounded rectangle.",
    "svg_star": "Draw a five-pointed gold star with an orange border.",
    "svg_smiley": "Draw a smiley face: a yellow circle with two black dot eyes, a curved smile, and optional eyebrows.",
    "svg_heart": "Draw a filled red heart shape as SVG: use a polygon with fill='red' and points='100,170 20,70 50,20 100,60 150,20 180,70'.",
    "svg_arrow": "Draw a large right-pointing arrow in blue.",
    "svg_play_button": "Draw a play button as SVG: a light-gray circle cx=100 cy=100 r=90 fill='#dcdcdc' stroke='#444'. Inside it, a solid dark-gray right-pointing triangle: polygon points='75,50 75,150 155,100' fill='#444' stroke='none'.",
    "svg_wifi_icon": (
        "Draw a WiFi icon as SVG. Use SVG path elements with arc commands (A), NOT startAngle/endAngle. "
        "Output exactly these elements inside a 200x200 SVG: "
        "1) <circle cx='100' cy='160' r='8' fill='#555'/> "
        "2) <path d='M 75,160 A30,30 0 0 1 125,160' fill='none' stroke='#555' stroke-width='5' stroke-linecap='round'/> "
        "3) <path d='M 50,160 A55,55 0 0 1 150,160' fill='none' stroke='#555' stroke-width='5' stroke-linecap='round'/> "
        "4) <path d='M 25,160 A80,80 0 0 1 175,160' fill='none' stroke='#555' stroke-width='5' stroke-linecap='round'/>"
    ),
    "svg_magnify": "Draw a magnifying glass: a circle outline in the upper-left, with a single diagonal line extending from the lower-right edge of the circle down to the lower-right corner as the handle.",
    "svg_lock": "Draw a padlock icon as SVG: dark body rect x=30 y=100 width=140 height=90 fill=#555; left leg rect x=55 y=40 width=30 height=70 fill=#999; right leg rect x=115 y=40 width=30 height=70 fill=#999. The two light gray leg rects above the dark body form a U-shaped shackle.",
    "svg_sun": "Draw a sun icon with a central yellow circle and 8 yellow rays radiating outward at equal angles. Use line elements for each ray: top, upper-right, right, lower-right, bottom, lower-left, left, upper-left.",
    "svg_envelope": "Draw an envelope icon: a white rectangle with a dark border. Draw a V-shape at the top using a polyline from the top-left corner, to the center of the top edge (midpoint but lower), to the top-right corner.",
    "svg_stopwatch": "Draw a stopwatch icon as SVG: a large circle outline for the face, a small rectangle on top of the circle for the button, and a short line from the center upward inside the circle for the hand.",
    "svg_bar_chart_icon": "Draw a simple bar chart icon with 3 vertical bars of different heights.",
    "svg_checkmark": "Draw a green checkmark (tick) symbol inside a green circle.",

    # -- Mermaid sequence diagrams --
    "mermaid_login_seq": (
        "Draw a sequence diagram showing a login flow: User submits credentials to Web App, "
        "Web App calls Auth Service, Auth Service queries the Database, Database returns user record, "
        "Auth Service returns a JWT token to Web App, Web App shows the dashboard to User."
    ),
    "mermaid_payment_seq": (
        "Draw a sequence diagram for an e-commerce payment flow: Customer clicks 'Pay' in Browser, "
        "Browser sends order to API Server, API Server calls Payment Gateway, "
        "Payment Gateway charges the card and returns success, API Server confirms the order, "
        "API Server sends confirmation email, Browser shows order confirmation to Customer."
    ),
    "mermaid_api_seq": (
        "Draw a sequence diagram showing how a mobile app loads a news feed: "
        "App requests /feed from API Gateway, API Gateway authenticates with Auth Service, "
        "Auth Service returns 200 OK, API Gateway queries Article Service, "
        "Article Service reads from Cache (cache miss), Article Service queries Database, "
        "Database returns articles, Article Service stores in Cache, "
        "API Gateway returns article list to App, App renders the feed."
    ),
    "mermaid_websocket_seq": (
        "Draw a sequence diagram for a WebSocket chat flow: "
        "Client connects to Chat Server via WebSocket, Server acknowledges connection, "
        "Client sends a chat message, Server broadcasts to all connected Clients, "
        "Another Client receives the message, Client disconnects."
    ),
    "mermaid_deploy_seq": (
        "Draw a sequence diagram of a CI/CD deployment pipeline: "
        "Developer pushes code to GitHub, GitHub triggers CI Runner, "
        "CI Runner runs tests, tests pass so CI Runner builds Docker image, "
        "CI Runner pushes image to Registry, CI Runner deploys to Kubernetes cluster, "
        "Kubernetes pulls image and starts pods, CI Runner notifies Slack with success."
    ),

    # -- Mermaid class diagrams --
    "mermaid_animals_class": (
        "Draw a UML class diagram for an animal hierarchy: Animal base class with name, age, and speak() method; "
        "Dog subclass with breed field and fetch() method; Cat subclass with indoor boolean and purr() method; "
        "Bird subclass with wingspan field and fly() method."
    ),
    "mermaid_ecommerce_class": (
        "Draw a UML class diagram for an e-commerce system: User class with userId, email, and placeOrder() method; "
        "Order class with orderId, date, totalAmount, and getStatus() method; "
        "Product class with productId, name, price, and getDescription() method; "
        "User has many Orders, Order contains many Products."
    ),
    "mermaid_bank_class": (
        "Draw a UML class diagram for a banking system: Account abstract class with accountNumber and balance fields, "
        "and deposit() and withdraw() methods; SavingsAccount subclass with interestRate field; "
        "CheckingAccount subclass with overdraftLimit field; Customer class with name and email, associated with Account."
    ),

    # -- Mermaid Gantt charts --
    "mermaid_software_gantt": (
        "Draw a Gantt chart for a software development project over 8 weeks: "
        "Week 1 is Requirements gathering; Weeks 2-3 are System Design; "
        "Weeks 3-6 are Backend Development (starting mid-week 3); "
        "Weeks 4-6 are Frontend Development; Week 7 is Integration Testing; Week 8 is Deployment."
    ),
    "mermaid_product_launch_gantt": (
        "Draw a Gantt chart for a product launch over 12 weeks: "
        "Weeks 1-2: Market Research; Weeks 2-4: Product Design; "
        "Weeks 4-8: Manufacturing; Weeks 6-9: Marketing Campaign; "
        "Weeks 9-11: Distribution Setup; Week 12: Launch."
    ),

    # -- Mermaid ER diagrams --
    "mermaid_library_er": (
        "Draw an entity-relationship diagram for a library management system: "
        "BOOK entity with isbn (PK), title, author, year; "
        "MEMBER entity with memberId (PK), name, email, joinDate; "
        "LOAN entity with loanId (PK), borrowDate, returnDate; "
        "A MEMBER can have many LOANs; a LOAN is for exactly one BOOK."
    ),
    "mermaid_hospital_er": (
        "Draw an entity-relationship diagram for a hospital: "
        "PATIENT entity with patientId, name, dateOfBirth; "
        "DOCTOR entity with doctorId, name, specialization; "
        "APPOINTMENT entity with appointmentId, date, time, diagnosis; "
        "Each APPOINTMENT involves one PATIENT and one DOCTOR."
    ),

    # -- Vega-Lite heatmaps --
    "vegalite_activity_heatmap": (
        "Create a heatmap showing website activity levels by day of week (rows: Mon-Sun) "
        "and time of day (columns: Morning, Afternoon, Evening, Night). "
        "Use realistic activity values: weekdays peak in the afternoon, weekends peak in the morning."
    ),
    "vegalite_temperature_heatmap": (
        "Create a heatmap of average monthly temperatures (°C) for five cities: "
        "New York, London, Tokyo, Sydney, Dubai. Rows are months (Jan through Dec), "
        "columns are cities. Use a red-blue diverging color scale where blue=cold, red=hot. "
        "Use realistic approximate temperature values."
    ),
    "vegalite_exam_scores_heatmap": (
        "Create a heatmap showing exam scores for 8 students across 5 subjects: "
        "Math, Physics, Chemistry, English, History. Each cell shows the score (0-100). "
        "Use a green color scale where darker=higher score."
    ),
    "vegalite_sales_heatmap": (
        "Create a heatmap of monthly sales revenue (in thousands) for 4 product categories "
        "(Electronics, Clothing, Food, Books) across the 4 quarters (Q1-Q4). "
        "Color by revenue amount with a yellow-orange-red scale."
    ),

    # -- Vega-Lite bubble charts --
    "vegalite_countries_bubble": (
        "Draw a bubble chart comparing 8 countries by GDP per capita (x-axis, in USD), "
        "life expectancy (y-axis, in years), and population (bubble size). "
        "Include: USA (63k, 79y, 330M), Germany (46k, 81y, 83M), Japan (40k, 84y, 125M), "
        "China (12k, 77y, 1400M), Brazil (8.8k, 75y, 213M), India (2.1k, 70y, 1380M), "
        "Nigeria (2k, 54y, 210M), Norway (80k, 83y, 5M). Color by country name."
    ),
    "vegalite_tech_companies_bubble": (
        "Draw a bubble chart of major tech companies: x-axis is revenue (billions USD), "
        "y-axis is employee count (thousands), bubble size is market cap (billions USD). "
        "Include: Apple (394B rev, 164k emp, 2800B cap), Microsoft (211B, 221k, 2400B), "
        "Google (283B, 182k, 1700B), Meta (116B, 77k, 800B), Amazon (514B, 1540k, 1600B), "
        "Netflix (33B, 13k, 160B). Color by company name."
    ),

    # -- Vega-Lite other complex charts --
    "vegalite_layered_bars_line": (
        "Create a layered chart with monthly sales data: "
        "bar chart showing monthly revenue (bars) overlaid with a line showing cumulative revenue. "
        "Months: Jan=80, Feb=95, Mar=70, Apr=110, May=90, Jun=120, Jul=85, Aug=105. "
        "Use blue bars and a red line."
    ),

    # -- HTML: document layouts, cards, tables --
    "html_invoice": (
        "Create an invoice for a freelance web developer: "
        "company name 'DevCraft Studio', client 'TechCorp Inc', invoice date 2024-03-15, "
        "line items: Frontend Development 60h @ $120/h = $7,200, "
        "API Integration 20h @ $100/h = $2,000, Testing 10h @ $80/h = $800. "
        "Subtotal $10,000, Tax 10% = $1,000, Total $11,000. Professional dark-header styling."
    ),
    "html_receipt": (
        "Design a restaurant receipt for 'The Golden Fork': "
        "order date 2024-06-10, table 7. Items: Caesar Salad $12.50, Grilled Salmon $28.00, "
        "Pasta Carbonara $18.00, Tiramisu $9.00, 2x House Wine $15.00 each. "
        "Subtotal $97.50, Tax (8%) $7.80, Tip (18%) $17.55, Total $122.85. "
        "Clean minimal styling with centered layout."
    ),
    "html_profile_card": (
        "Create a profile card for a machine learning engineer: "
        "name 'Alex Rivera', role 'ML Engineer', company 'DataFlow AI'. "
        "Stats: 142 models trained, 97.3% best accuracy, 5 years experience. "
        "Skills: Python, PyTorch, Kubernetes. Blue accent color, card with shadow."
    ),
    "html_comparison_table": (
        "Make a feature comparison table for three project management tools: "
        "Jira, Trello, Asana. Rows: Price (Free/$10/$13 per user), "
        "Kanban boards (Yes/Yes/Yes), Gantt charts (Yes/No/Yes), "
        "Time tracking (Yes/No/Yes), API access (Yes/Yes/Yes), "
        "Mobile app (Yes/Yes/Yes), Integrations (3000+/200+/400+). "
        "Blue header row, alternating gray/white rows."
    ),
    "html_nutrition_label": (
        "Create an HTML nutrition facts label similar to a US food product label. "
        "Product: 'Whole Grain Oatmeal'. Serving size 40g, Calories 150, "
        "Total Fat 3g (4% DV), Saturated Fat 0.5g, Sodium 0mg, "
        "Total Carb 27g (10% DV), Fiber 4g (14% DV), Sugars 1g, Protein 5g, "
        "Vitamin D 0%, Calcium 2%, Iron 10%. Black border, thick dividers."
    ),
    "html_course_schedule": (
        "Create a weekly class schedule table for a college student. "
        "Columns: Time, Monday, Tuesday, Wednesday, Thursday, Friday. "
        "Rows: 9am (Math/--/Math/--/Math), 11am (Physics/CS101/Physics/CS101/--), "
        "2pm (--/Lab/--/Lab/--), 4pm (English/--/English/--/English). "
        "Color each subject differently, use light pastel fills."
    ),
}
