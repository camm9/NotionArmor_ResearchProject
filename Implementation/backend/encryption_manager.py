from cryptography.fernet import Fernet
from dotenv import load_dotenv, find_dotenv, set_key
import os

class EncryptionManager:
    """ Encryption Manager for storing API key in browser session storage
    Check if .env file exists with encyrption key. If not create one and store new
    encryption key.
    """
    _instance = None
    _key = None
    _fernet = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EncryptionManager, cls).__new__(cls)

        env_path = find_dotenv()

        if not env_path:
            #create new .env file with key
            cls._create_env_with_key()
            load_dotenv()
        else:
            load_dotenv()
            encryption_key = os.getenv("ENCRYPTION_KEY")
            if not encryption_key:
                # add encryption key to .env
                cls._add_encryption_key_to_existing_env(env_path)
                load_dotenv()

        encryption_key = os.getenv("ENCRYPTION_KEY")
        encryption_key = encryption_key.strip("'").strip('"') #strip quotations

        # set up Fernet with the retrieved key
        try:
            cls._key = encryption_key.encode()
            cls._fernet = Fernet(cls._key)
            print(">>> Encryption manager initialized.")
        except Exception as e:
            print(f">>> Error creating encryption: {e}")
            print(f">>> Please delete {env_path} and re-run this command.")
        return cls._instance


    @classmethod
    def _create_env_with_key(cls):
        """ Create an env file with an encryption key"""
        key = Fernet.generate_key().decode()

        with open(".env", "w") as f:
            f.write(f"ENCRYPTION_KEY={key}\n")

        print(">>> Created .env file with encryption key.")

    @classmethod
    def _add_encryption_key_to_existing_env(cls, env_path):
        """ If .env already exists but does not contain encryption key,
            create a new one and store new encryption key.
        """
        key = Fernet.generate_key().decode()

        set_key(env_path,"ENCRYPTION_KEY", key)

        print(">>> Added encryption key to .env file.")


    def encrypt_data(self, data):
        """ Encrypt data using stored encryption key in .env.
        Data needs to be a string, this will return bytes
        """

        return self._fernet.encrypt(data.encode())

    def decrypt_data(self, data):
        """ Decrypt data using stored encryption key in .env
            This takes bytes as input and returns string
        """
        return self._fernet.decrypt(data)


