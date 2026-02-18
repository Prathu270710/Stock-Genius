class BoardSelector:
    """Handle board and stock selection"""
    
    BOARDS = {
        'NSE': {
            'name': 'National Stock Exchange (India)',
            'suffix': '.NS',
            'popular_stocks': [
                'RELIANCE', 'TCS', 'INFY', 'WIPRO', 'BAJAJ-AUTO',
                'HDFC', 'ICICIBANK', 'AXISBANK', 'HDFCBANK',
                'MARUTI', 'SUNPHARMA', 'ASIANPAINT', 'ITC', 'IOC'
            ]
        },
        'BSE': {
            'name': 'Bombay Stock Exchange (India)',
            'suffix': '.BO',
            'popular_stocks': [
                'RELIANCE', 'TCS', 'INFY', 'WIPRO', 'BAJAJ-AUTO'
            ]
        },
        'NASDAQ': {
            'name': 'NASDAQ (USA Tech)',
            'suffix': '',
            'popular_stocks': [
                'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA',
                'META', 'AMZN', 'NFLX', 'PYPL', 'ASML'
            ]
        },
        'NYSE': {
            'name': 'New York Stock Exchange (USA)',
            'suffix': '',
            'popular_stocks': [
                'JPM', 'BAC', 'WFC', 'GS', 'MS',
                'PG', 'KO', 'MCD', 'PEP', 'JNJ'
            ]
        },
        'LSE': {
            'name': 'London Stock Exchange (UK)',
            'suffix': '.L',
            'popular_stocks': [
                'HSBA', 'BARR', 'LLOY', 'BARC', 'GSK'
            ]
        }
    }
    
    @staticmethod
    def display_boards():
        """Show available boards"""
        print("\n" + "="*80)
        print("SELECT STOCK EXCHANGE / BOARD")
        print("="*80 + "\n")
        
        for idx, (code, info) in enumerate(BoardSelector.BOARDS.items(), 1):
            print(f"{idx}. {code:10} - {info['name']}")
        
        print(f"{len(BoardSelector.BOARDS) + 1}. OTHER     - Enter custom ticker\n")
    
    @staticmethod
    def get_board_selection():
        """Get user board selection"""
        BoardSelector.display_boards()
        
        while True:
            choice = input("Enter choice (1-6): ").strip()
            
            boards = list(BoardSelector.BOARDS.keys())
            
            if choice == '6':
                return 'OTHER', ''
            elif choice in ['1', '2', '3', '4', '5']:
                board_code = boards[int(choice) - 1]
                return board_code, BoardSelector.BOARDS[board_code]['suffix']
            else:
                print("Invalid choice. Please try again.\n")
    
    @staticmethod
    def display_popular_stocks(board_code):
        """Show popular stocks for selected board"""
        if board_code not in BoardSelector.BOARDS:
            return []
        
        stocks = BoardSelector.BOARDS[board_code]['popular_stocks']
        
        print("\n" + "="*80)
        print(f"POPULAR STOCKS - {board_code}")
        print("="*80 + "\n")
        
        for idx, stock in enumerate(stocks, 1):
            print(f"{idx:2}. {stock}")
        
        print(f"{len(stocks) + 1}. Search for different stock\n")
        
        return stocks
    
    @staticmethod
    def get_stock_selection(board_code, suffix):
        """Get user stock selection - Allow typing custom stock names"""
        stocks = BoardSelector.display_popular_stocks(board_code)
        
        while True:
            choice = input("Enter choice (or stock name): ").strip().upper()
            
            # Check if user selected from list
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(stocks):
                    # User selected from popular list
                    ticker = stocks[idx] + suffix
                    return ticker
                elif choice == str(len(stocks) + 1):
                    # User selected "Search for different stock"
                    print()
                    custom_stock = input("Enter stock name (e.g., RELIANCE, TESLA, ORACLE): ").strip().upper()
                    
                    if custom_stock:
                        # Add suffix if needed
                        if board_code in ['NSE', 'BSE', 'LSE'] and not custom_stock.endswith(suffix):
                            ticker = custom_stock + suffix
                        else:
                            ticker = custom_stock
                        
                        print(f"✓ Selected: {ticker}\n")
                        return ticker
                    else:
                        print("❌ Please enter a valid stock name.\n")
                        continue
                else:
                    print("❌ Invalid choice. Try again.\n")
            else:
                # User typed stock name directly (instead of selecting number)
                if choice:
                    # Add suffix if needed
                    if board_code in ['NSE', 'BSE', 'LSE'] and not choice.endswith(suffix):
                        ticker = choice + suffix
                    else:
                        ticker = choice
                    
                    print(f"✓ Selected: {ticker}\n")
                    return ticker
                else:
                    print("❌ Please enter a valid choice or stock name.\n")
    
    @staticmethod
    def get_ticker_from_user():
        """Complete flow: Board -> Stock"""
        board_code, suffix = BoardSelector.get_board_selection()
        
        if board_code == 'OTHER':
            print()
            ticker = input("Enter stock ticker (e.g., ONGC.NS, ORACLE, RELIANCE.NS): ").strip().upper()
            if ticker:
                print(f"✓ Selected: {ticker}\n")
                return ticker
            else:
                print("❌ No ticker entered.\n")
                return None
        else:
            ticker = BoardSelector.get_stock_selection(board_code, suffix)
            return ticker


if __name__ == "__main__":
    # Test the selector
    ticker = BoardSelector.get_ticker_from_user()
    if ticker:
        print(f"✅ Selected: {ticker}\n")
    else:
        print("No selection made.\n")
