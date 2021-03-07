import * as crypto from 'crypto';

// transfer of funds between two wallets
class Transaction {
    constructor(
        public amount: number,
        public payer: string,
        public payee: string
    ) {}

    toString() {
        return JSON.stringify(this)
    }
}

// individual block on the chain
class Block {
    public nonce = Math.round(Math.random() * 999999999)

    constructor(
        public prevHash: string,
        public transaction: Transaction,
        public ts = Date.now()
    ) {}

    get hash() {
        const str = JSON.stringify(this);
        const hash = crypto.createHash('SHA256');

        hash.update(str).end();

        return hash.digest('hex');
    }
}

// blockchain
class BlockChain {
    // singleton instance
    public static instance = new BlockChain();

    blockchain: Block[];

    constructor() {
        this.blockchain = [
            // genesis block
            new Block('', new Transaction(100, 'genesis', 'satoshi'))
        ];
    }

    // most recent block
    get lastBlock() {
        return this.blockchain[this.blockchain.length - 1];
    }

    // proof of work system
    mine(nonce: number) {
        let solution = 1;
        console.log('⛏️  mining...')

        while (true) {
            const hash = crypto.createHash('MD5');
            hash.update((nonce + solution).toString()).end();

            const attempt = hash.digest('hex');

            if (attempt.substr(0, 4) == '0000') {
                console.log('Solved: ${solution');

                return solution;
            }

            solution += 1;
        }
    }

    // add a new block to the chain if valid signature&proof of work is complete
    addBlock(transaction: Transaction, payerPublicKey: string, signature: Buffer) {
        const verify = crypto.createVerify('SHA256');
        verify.update(transaction.toString());

        const isValid = verify.verify(payerPublicKey, signature);

        if (isValid) {
            const newBlock = new Block(this.lastBlock.hash, transaction);
            this.mine(newBlock.nonce);
            this.blockchain.push(newBlock);
        }
    }
}

// wallet gives a user a public/private keypair
class Wallet {
    public publicKey: string;
    public privateKey: string;

    constructor() {
        const keypair = crypto.generateKeyPairSync('rsa', {
            modulusLength: 2048,
            publicKeyEncoding: {type: 'spki', format: 'pem'},
            privateKeyEncoding: {type: 'pkcs8', format: 'pem'},
        });

        this.privateKey = keypair.privateKey;
        this.publicKey = keypair.publicKey;
    }

    pay(amount: number, payeePublicKey: string) {
        const transaction = new Transaction(amount, this.publicKey, payeePublicKey);
        const sign = crypto.createSign('SHA256');
        sign.update(transaction.toString()).end();

        const signature = sign.sign(this.privateKey);

        BlockChain.instance.addBlock(transaction, this.publicKey, signature);
    }
}

const satoshi = new Wallet();
const bob = new Wallet();
const alice = new Wallet();

satoshi.pay(50, bob.publicKey);
bob.pay(23, alice.publicKey);
alice.pay(5, bob.publicKey);

console.log(BlockChain.instance)