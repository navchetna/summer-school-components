from main import get_paragraphs

def main():
    # List of example URLs
    url_list = [
        "https://en.wiktionary.org/wiki/Wiktionary:Main_Page",
    ]
    
    # Extract paragraphs from the URLs
    paragraphs = get_paragraphs(url_list)
    
    # Print the extracted paragraphs
    for p in paragraphs:
        print(p)

if __name__ == "__main__":
    main()
