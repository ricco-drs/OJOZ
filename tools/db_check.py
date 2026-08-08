# utils/db_check.py
"""
Script de utilidad para verificar el contenido de la base de datos.
Ejecutar con: python -m app.utils.db_check
"""

from app.db.engine import init_db
from app.db import dao

def main():
    print("=== Verificación de Base de Datos ===\n")
    
    # Inicializar BD si no existe
    init_db()
    
    # Obtener todos los usuarios
    users = dao.get_all_users()
    
    if not users:
        print("[!] No hay usuarios registrados en la base de datos.\n")
        print("Registra un usuario usando la opción 1 del asistente.\n")
        return
    
    print(f"[OK] Usuarios registrados: {len(users)}\n")
    
    for user in users:
        print(f"  • ID: {user['id']}")
        print(f"    Nombre: {user['name']}")
        print(f"    Alias: {user.get('alias', 'N/A')}")
        print(f"    Registrado: {user['created_at']}")
        print()

if __name__ == "__main__":
    main()
