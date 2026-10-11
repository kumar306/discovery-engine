import asyncio
import sys

# psycopg async mode cannot use the default windows event loop
# linux is where this ships and it is fine there
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
