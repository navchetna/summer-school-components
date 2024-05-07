import sys
from adapter import adapter


def main():
    adapter_obj = adapter(chunk_size=40, chunk_overlap=4, seperator=",")
    document = [
        """Initial competition came from the smaller trijet widebodies: the Lockheed L-1011 (introduced
in 1972),  McDonnell Douglas DC-10 (1971) and later MD-11 (1990). Airbus competed with
later variants with the heaviest versions of the A340 until surpassing the 747 in size with the
A380, delivered between 2007 and 2021. Freighter variants of the 747 remain popular with
cargo airlines. The final 747 was delivered to Atlas Air in January 2023 after a 54-year
production run, with 1,574 aircraft built. As of January 2023, 64 Boeing 747s (4.1%) have
been lost in accidents and incidents, in which a total of 3,746 people have died."""
   ]
    chunks = adapter_obj.get_chunks(document)
    print("Chunks: ")
    print(chunks)


if __name__ == "__main__":
    main()
