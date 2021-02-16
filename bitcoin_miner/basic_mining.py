from hashlib import sha256
MAX_NONCE = 100000000000

import time

def SHA256(text):
    return sha256(text.encode("ascii")).hexdigest()

def mining(block_num, transaction, prev_hash, prefix_zeros):
    prefix_str = '0' * prefix_zeros

    for nonce in range(MAX_NONCE):
        text = str(block_num) + transaction + prev_hash + str(nonce)
        new_hash = SHA256(text)

        if new_hash.startswith(prefix_str):
            print("Successfully mined bitcoins with nonce value:{nonce}")
            return new_hash

if __name__ == '__main__':
    transactions = '''
    John->Ben->20,
    Max->Cara->45
    '''
    difficulty = 4 # this can be changed

    start = time.time()
    print("start mining")

    new_hash = mining(5,transactions,'0000000xa036944e29568d0cff17edbe038f81208fecf9a66be9a2b8321c6ec7', difficulty)
    total_time = str((time.time() - start))
    
    print(f"end mining. Mining took: {total_time} seconds")
    print(new_hash)